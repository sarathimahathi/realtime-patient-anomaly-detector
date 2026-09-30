"""
Unit and integration tests for patient telemetry streamer, API endpoints, and WebSockets.
"""

import asyncio
import pytest
from fastapi.testclient import TestClient

from src.data_loader import (
    EXPECTED_COLUMNS,
    load_kaggle_dataset,
    simulate_patient_stream,
)
from src.server import app


def test_load_kaggle_dataset_fallback():
    """Verify fallback dataset generation and expected columns."""
    df = load_kaggle_dataset("data/raw/test_non_existent.csv")
    assert not df.empty
    assert list(df.columns) == EXPECTED_COLUMNS
    assert len(df) >= 1000


@pytest.mark.asyncio
async def test_simulate_patient_stream():
    """Verify async telemetry generator yields valid vital records."""
    gen = simulate_patient_stream(patient_id="PATIENT-TEST-01", interval=0.01, anomaly_rate=0.0)
    samples = []
    async for reading in gen:
        samples.append(reading)
        if len(samples) >= 5:
            break

    assert len(samples) == 5
    for s in samples:
        assert s["patient_id"] == "PATIENT-TEST-01"
        assert 40.0 <= s["heart_rate"] <= 220.0
        assert 50.0 <= s["spo2"] <= 100.0
        assert 50.0 <= s["systolic_bp"] <= 250.0
        assert 30.0 <= s["diastolic_bp"] <= 150.0
        assert 5.0 <= s["respiratory_rate"] <= 50.0
        assert 34.0 <= s["temperature"] <= 43.0


@pytest.mark.asyncio
async def test_simulate_patient_stream_anomaly_injection():
    """Verify anomalous readings are produced when anomaly_rate=1.0."""
    gen = simulate_patient_stream(patient_id="PATIENT-TEST-02", interval=0.01, anomaly_rate=1.0)
    samples = []
    async for reading in gen:
        samples.append(reading)
        if len(samples) >= 10:
            break

    # At least some readings should have extreme values (e.g. spo2 < 90 or HR > 130 or SBP > 170)
    has_extreme = any(
        s["spo2"] < 90.0 or s["heart_rate"] > 130.0 or s["systolic_bp"] > 170.0 or s["systolic_bp"] < 85.0
        for s in samples
    )
    assert has_extreme, "Expected at least one extreme reading when anomaly_rate=1.0"


def test_api_health_endpoint():
    """Test health check route."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "online"
        assert "model_loaded" in data
        assert "scaler_loaded" in data


def test_api_v1_predict_endpoint_normal():
    """Test synchronous prediction endpoint with normal vitals."""
    with TestClient(app) as client:
        payload = {
            "patient_id": "PATIENT-NORMAL-01",
            "heart_rate": 72.0,
            "spo2": 98.5,
            "systolic_bp": 120.0,
            "diastolic_bp": 80.0,
            "respiratory_rate": 16.0,
            "temperature": 37.0,
        }
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["patient_id"] == "PATIENT-NORMAL-01"
        assert data["is_anomaly"] is False
        assert data["severity"] == "NORMAL"
        assert data["anomaly_score"] < 0.50
        assert "derived_metrics" in data
        assert data["derived_metrics"]["pulse_pressure"] == 40.0
        assert round(data["derived_metrics"]["mean_arterial_pressure"], 1) == 93.3


def test_api_v1_predict_endpoint_critical_guardrails():
    """Test deterministic guardrail triggers (e.g., hypoxia + severe tachycardia)."""
    with TestClient(app) as client:
        payload = {
            "patient_id": "PATIENT-CRIT-01",
            "heart_rate": 145.0,  # HR > 130 bpm
            "spo2": 85.0,         # SpO2 < 90%
            "systolic_bp": 72.0,  # SBP < 85 mmHg
            "diastolic_bp": 45.0,
            "respiratory_rate": 28.0,
            "temperature": 39.5,  # Temp > 39.0°C
        }
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["patient_id"] == "PATIENT-CRIT-01"
        assert data["is_anomaly"] is True
        assert data["severity"] == "CRITICAL"
        assert data["anomaly_score"] >= 0.90
        assert len(data["flagged_reasons"]) >= 3


def test_websocket_live_endpoint_schema():
    """Verify WebSocket endpoint emits valid JSON schema matching all requirements."""
    with TestClient(app) as client:
        with client.websocket_connect("/ws/live/PATIENT-WS-TEST?interval=0.05") as ws:
            # Receive first telemetry frame
            msg = ws.receive_json()

            # 1. Top-level keys verification
            required_top_keys = {"raw_vitals", "is_anomaly", "severity", "anomaly_score", "timestamp", "flagged_reasons"}
            assert required_top_keys.issubset(set(msg.keys())), f"Missing keys in payload: {set(msg.keys())}"

            # 2. Raw vitals sub-schema
            vitals = msg["raw_vitals"]
            for col in EXPECTED_COLUMNS:
                assert col in vitals, f"Missing vital field: '{col}'"

            # 3. Types and ranges
            assert isinstance(msg["is_anomaly"], bool)
            assert msg["severity"] in ("NORMAL", "WARNING", "CRITICAL")
            assert isinstance(msg["anomaly_score"], (int, float))
            assert 0.0 <= msg["anomaly_score"] <= 1.0
            assert isinstance(msg["flagged_reasons"], list)
