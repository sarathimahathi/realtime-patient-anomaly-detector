"""
FastAPI production application exposing REST inference and WebSocket telemetry streaming.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.data_loader import simulate_patient_stream
from src.model import PatientAnomalyDetector
from src.preprocessor import VitalSignsPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("patient-anomaly-server")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "anomaly_detector.joblib"
SCALER_PATH = BASE_DIR / "models" / "scaler.joblib"
DASHBOARD_DIR = BASE_DIR / "dashboard"

# Global loaded artifacts
detector_instance: Optional[PatientAnomalyDetector] = None
preprocessor_instance: Optional[VitalSignsPreprocessor] = None


def load_artifacts() -> None:
    """Load trained model and scaler artifacts from disk."""
    global detector_instance, preprocessor_instance

    if not SCALER_PATH.exists() or not MODEL_PATH.exists():
        logger.warning("Artifacts missing in models/. Auto-running training script...")
        import subprocess
        import sys
        subprocess.run([sys.executable, "train.py"], cwd=str(BASE_DIR), check=True)

    try:
        preprocessor_instance = VitalSignsPreprocessor.load(SCALER_PATH)
        detector_instance = PatientAnomalyDetector.load(MODEL_PATH)
        detector_instance.preprocessor = preprocessor_instance
        logger.info("Successfully loaded model and scaler artifacts.")
    except Exception as exc:
        logger.error(f"Failed to load artifacts: {exc}", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup loading and shutdown cleanup."""
    load_artifacts()
    yield


app = FastAPI(
    title="Realtime Patient Anomaly Detector",
    description="ICU patient telemetry streaming and real-time anomaly detection with ML & clinical guardrails.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount dashboard static assets
if DASHBOARD_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(DASHBOARD_DIR)), name="static")


# Pydantic Schemas
class VitalReadingInput(BaseModel):
    patient_id: str = Field(default="PATIENT-1001", description="Patient identifier")
    timestamp: Optional[float] = Field(default=None, description="Epoch timestamp of telemetry reading")
    heart_rate: float = Field(..., ge=20.0, le=280.0, description="Heart rate (BPM)")
    spo2: float = Field(..., ge=40.0, le=100.0, description="Oxygen saturation SpO2 (%)")
    systolic_bp: float = Field(..., ge=30.0, le=300.0, description="Systolic blood pressure (mmHg)")
    diastolic_bp: float = Field(..., ge=20.0, le=200.0, description="Diastolic blood pressure (mmHg)")
    respiratory_rate: float = Field(..., ge=4.0, le=70.0, description="Respiratory rate (breaths/min)")
    temperature: float = Field(..., ge=32.0, le=44.0, description="Body temperature (°C)")


class AnomalyPredictionResponse(BaseModel):
    patient_id: str
    is_anomaly: bool
    anomaly_score: float
    severity: str
    flagged_reasons: List[str]
    derived_metrics: Dict[str, float]


@app.get("/health", tags=["System"])
def health_check() -> Dict[str, Any]:
    """Returns system status and model load status."""
    is_model_loaded = detector_instance is not None and detector_instance.is_fitted
    is_scaler_loaded = preprocessor_instance is not None and preprocessor_instance.is_fitted
    return {
        "status": "online",
        "model_loaded": is_model_loaded,
        "scaler_loaded": is_scaler_loaded,
        "model_path": str(MODEL_PATH),
        "scaler_path": str(SCALER_PATH),
    }


@app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
@app.get("/dashboard", response_class=HTMLResponse, tags=["Dashboard"])
def get_dashboard() -> Any:
    """Serve the real-time patient telemetry web visualizer."""
    index_file = DASHBOARD_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return HTMLResponse("<h2>Patient Telemetry Dashboard not found.</h2>")



@app.post("/api/v1/predict", response_model=AnomalyPredictionResponse, tags=["Inference"])
def predict_vitals(reading: VitalReadingInput) -> AnomalyPredictionResponse:
    """Synchronous endpoint taking a single JSON reading and returning anomaly status."""
    if detector_instance is None or not detector_instance.is_fitted:
        raise HTTPException(status_code=503, detail="Anomaly detector model is not ready.")

    vitals_dict = reading.model_dump()
    result = detector_instance.predict(vitals_dict)

    pulse_pressure = reading.systolic_bp - reading.diastolic_bp
    map_score = reading.diastolic_bp + (pulse_pressure / 3.0)

    return AnomalyPredictionResponse(
        patient_id=reading.patient_id,
        is_anomaly=result["is_anomaly"],
        anomaly_score=result["anomaly_score"],
        severity=result["severity"],
        flagged_reasons=result["flagged_reasons"],
        derived_metrics={
            "pulse_pressure": round(pulse_pressure, 2),
            "mean_arterial_pressure": round(map_score, 2),
            "shock_index": round(reading.heart_rate / max(reading.systolic_bp, 1.0), 2),
        },
    )


@app.websocket("/ws/live/{patient_id}")
async def websocket_live_stream(
    websocket: WebSocket,
    patient_id: str,
    interval: float = 1.0,
    anomaly_rate: float = 0.10,
) -> None:
    """WebSocket streaming endpoint for continuous patient telemetry and real-time inference.

    Emits payload every 1 second:
    {
        "raw_vitals": { ... },
        "is_anomaly": bool,
        "severity": "NORMAL" | "WARNING" | "CRITICAL",
        "anomaly_score": float,
        "timestamp": float,
        "flagged_reasons": [ ... ]
    }
    """
    await websocket.accept()
    logger.info(f"Client connected to live telemetry stream for patient {patient_id}")

    stream_gen = simulate_patient_stream(
        patient_id=patient_id,
        interval=interval,
        anomaly_rate=anomaly_rate,
    )

    # Exponential smoothing window for anomaly score
    smoothed_score: Optional[float] = None
    alpha = 0.4  # smoothing factor

    try:
        async for vitals in stream_gen:
            # Check for optional incoming control message (non-blocking)
            try:
                client_msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                parsed = json.loads(client_msg)
                # Client can inject an acute emergency override
                if parsed.get("inject_anomaly"):
                    vitals["spo2"] = 84.0
                    vitals["heart_rate"] = 155.0
                    vitals["systolic_bp"] = 72.0
            except (asyncio.TimeoutError, json.JSONDecodeError):
                pass

            # Run inference
            if detector_instance:
                pred = detector_instance.predict(vitals)
                is_anomaly = pred["is_anomaly"]
                severity = pred["severity"]
                raw_score = pred["anomaly_score"]
                flagged_reasons = pred["flagged_reasons"]
            else:
                is_anomaly = False
                severity = "NORMAL"
                raw_score = 0.0
                flagged_reasons = []

            # Compute smoothed anomaly score
            if smoothed_score is None:
                smoothed_score = raw_score
            else:
                smoothed_score = (alpha * raw_score) + ((1.0 - alpha) * smoothed_score)

            payload = {
                "raw_vitals": vitals,
                "is_anomaly": is_anomaly,
                "severity": severity,
                "anomaly_score": round(smoothed_score, 4),
                "timestamp": vitals.get("timestamp"),
                "flagged_reasons": flagged_reasons,
            }

            await websocket.send_json(payload)

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected for patient: {patient_id}")
    except Exception as exc:
        logger.error(f"WebSocket stream error for patient {patient_id}: {exc}", exc_info=True)
        await websocket.close()
