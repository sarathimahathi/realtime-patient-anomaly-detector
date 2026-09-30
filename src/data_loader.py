"""
Data loader and real-time streaming telemetry simulator for patient vital signs.
Supports automated ingestion of Kaggle/Excel human vital signs datasets.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS: List[str] = [
    "patient_id",
    "timestamp",
    "heart_rate",
    "spo2",
    "systolic_bp",
    "diastolic_bp",
    "respiratory_rate",
    "temperature",
]

# Standard column mappings for Kaggle/clinical datasets
COLUMN_ALIASES: Dict[str, str] = {
    "patient id": "patient_id",
    "patient_id": "patient_id",
    "id": "patient_id",
    "heart rate": "heart_rate",
    "heart_rate": "heart_rate",
    "hr": "heart_rate",
    "pulse": "heart_rate",
    "oxygen saturation": "spo2",
    "oxygen_saturation": "spo2",
    "spo2": "spo2",
    "o2": "spo2",
    "systolic blood pressure": "systolic_bp",
    "systolic_bp": "systolic_bp",
    "systolic": "systolic_bp",
    "sbp": "systolic_bp",
    "diastolic blood pressure": "diastolic_bp",
    "diastolic_bp": "diastolic_bp",
    "diastolic": "diastolic_bp",
    "dbp": "diastolic_bp",
    "respiratory rate": "respiratory_rate",
    "respiratory_rate": "respiratory_rate",
    "rr": "respiratory_rate",
    "body temperature": "temperature",
    "temperature": "temperature",
    "temp": "temperature",
    "timestamp": "timestamp",
    "time": "timestamp",
}

# Cache for real dataset records for replay streaming
_CACHED_STREAM_RECORDS: Optional[List[Dict[str, Any]]] = None


def normalize_vital_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize disparate naming conventions from Kaggle and clinical exports."""
    rename_dict: Dict[str, str] = {}
    for col in df.columns:
        norm = str(col).strip().lower()
        if norm in COLUMN_ALIASES:
            rename_dict[col] = COLUMN_ALIASES[norm]

    standardized = df.rename(columns=rename_dict)
    
    # Format patient_id
    if "patient_id" in standardized.columns:
        standardized["patient_id"] = standardized["patient_id"].apply(
            lambda x: f"PATIENT-{int(x):04d}" if str(x).isdigit() else str(x)
        )
    else:
        standardized["patient_id"] = "PATIENT-0001"

    # Convert timestamps to numeric seconds epoch
    if "timestamp" in standardized.columns:
        try:
            standardized["timestamp"] = pd.to_datetime(standardized["timestamp"]).astype("int64") / 1e9
        except Exception:
            standardized["timestamp"] = time.time()
    else:
        standardized["timestamp"] = time.time()

    return standardized


def generate_synthetic_vitals(
    n_samples: int = 1500,
    patient_ids: Optional[List[str]] = None,
    anomaly_rate: float = 0.05,
) -> pd.DataFrame:
    """Generate realistic synthetic vital signs DataFrame matching ICU distributions."""
    if patient_ids is None:
        patient_ids = [f"PATIENT-{1000 + i}" for i in range(1, 6)]

    base_time = time.time() - (n_samples * 2)
    records: List[Dict[str, Any]] = []

    for i in range(n_samples):
        pid = random.choice(patient_ids)
        ts = base_time + (i * 2)

        hr = float(np.clip(np.random.normal(75.0, 6.0), 55.0, 100.0))
        spo2 = float(np.clip(np.random.normal(98.0, 1.0), 95.0, 100.0))
        sbp = float(np.clip(np.random.normal(120.0, 8.0), 100.0, 138.0))
        dbp = float(np.clip(np.random.normal(78.0, 5.0), 65.0, 88.0))
        rr = float(np.clip(np.random.normal(16.0, 2.0), 12.0, 20.0))
        temp = float(np.clip(np.random.normal(37.0, 0.25), 36.5, 37.5))

        if random.random() < anomaly_rate:
            anomaly_kind = random.choice(["hypoxia", "tachycardia", "hypertension", "hypotension", "hyperthermia"])
            if anomaly_kind == "hypoxia":
                spo2 = float(random.uniform(74.0, 87.5))
                rr = float(random.uniform(24.0, 34.0))
            elif anomaly_kind == "tachycardia":
                hr = float(random.uniform(142.0, 178.0))
            elif anomaly_kind == "hypertension":
                sbp = float(random.uniform(182.0, 220.0))
                dbp = float(random.uniform(105.0, 130.0))
            elif anomaly_kind == "hypotension":
                sbp = float(random.uniform(65.0, 82.0))
                dbp = float(random.uniform(40.0, 52.0))
                hr = float(random.uniform(110.0, 135.0))
            elif anomaly_kind == "hyperthermia":
                temp = float(random.uniform(39.2, 40.8))
                hr = float(random.uniform(105.0, 125.0))

        records.append({
            "patient_id": pid,
            "timestamp": round(ts, 2),
            "heart_rate": round(hr, 1),
            "spo2": round(spo2, 1),
            "systolic_bp": round(sbp, 1),
            "diastolic_bp": round(dbp, 1),
            "respiratory_rate": round(rr, 1),
            "temperature": round(temp, 2),
        })

    return pd.DataFrame(records)[EXPECTED_COLUMNS]


def load_kaggle_dataset(file_path: str = "data/raw/patient_vitals.csv") -> pd.DataFrame:
    """Load patient vitals dataset from the uploaded dataset, CSV, or auto-generate fallback.

    Searches multiple potential locations:
    1. Specified file_path
    2. data/raw/human_vital_signs_dataset_2024.csv
    3. ../archive (1)/human_vital_signs_dataset_2024.csv
    """
    global _CACHED_STREAM_RECORDS

    candidate_paths: List[Path] = [
        Path(file_path),
        Path("data/raw/human_vital_signs_dataset_2024.csv"),
        Path("../archive (1)/human_vital_signs_dataset_2024.csv"),
        Path("c:/Users/ASUS/Desktop/New DSA/archive (1)/human_vital_signs_dataset_2024.csv"),
        Path("data/raw/patient_vitals.csv"),
    ]

    resolved_path: Optional[Path] = None

    # If the user explicitly passed a path that exists, use it
    target = Path(file_path)
    if target.exists() and target.is_file():
        resolved_path = target
    else:
        # Search candidate paths
        for candidate in candidate_paths:
            if candidate.exists() and candidate.is_file():
                resolved_path = candidate
                break

    if resolved_path:
        logger.info(f"Loading patient vitals from: {resolved_path.resolve()}")
        raw_df = pd.read_csv(resolved_path)
        standardized_df = normalize_vital_columns(raw_df)

        missing_cols = set(EXPECTED_COLUMNS) - set(standardized_df.columns)
        if missing_cols:
            raise ValueError(f"Dataset at {resolved_path} is missing expected columns: {missing_cols}")

        clean_df = standardized_df[EXPECTED_COLUMNS].copy()

        # If loading from raw uploaded dataset, save a normalized copy for fast access
        target_default = Path("data/raw/patient_vitals.csv")
        if resolved_path != target_default and not target_default.exists():
            try:
                target_default.parent.mkdir(parents=True, exist_ok=True)
                clean_df.head(25000).to_csv(target_default, index=False)
                logger.info(f"Cached normalized dataset slice to: {target_default.resolve()}")
            except Exception as exc:
                logger.debug(f"Could not cache dataset slice: {exc}")

        _CACHED_STREAM_RECORDS = clean_df.to_dict(orient="records")
        return clean_df

    # Fallback if none found
    logger.warning(f"File not found at '{file_path}'. Auto-generating realistic synthetic telemetry fallback.")
    df_synthetic = generate_synthetic_vitals(n_samples=2000, anomaly_rate=0.05)

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        df_synthetic.to_csv(target, index=False)
        logger.info(f"Persisted synthetic fallback dataset to: {target.resolve()}")
    except Exception as exc:
        logger.debug(f"Could not persist fallback dataset to disk: {exc}")

    _CACHED_STREAM_RECORDS = df_synthetic.to_dict(orient="records")
    return df_synthetic


async def simulate_patient_stream(
    patient_id: str,
    interval: float = 1.0,
    anomaly_rate: float = 0.1,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Asynchronously stream live patient vitals at regular intervals with injected anomalies.

    Replays telemetry from the uploaded real dataset or physiological baseline.
    """
    global _CACHED_STREAM_RECORDS

    if _CACHED_STREAM_RECORDS is None or len(_CACHED_STREAM_RECORDS) == 0:
        try:
            load_kaggle_dataset()
        except Exception:
            _CACHED_STREAM_RECORDS = []

    record_idx = 0
    records_count = len(_CACHED_STREAM_RECORDS) if _CACHED_STREAM_RECORDS else 0

    while True:
        now = time.time()
        is_anomaly = random.random() < anomaly_rate

        if not is_anomaly:
            if records_count > 0:
                base_record = _CACHED_STREAM_RECORDS[record_idx % records_count]
                record_idx += 1
                hr = float(base_record["heart_rate"])
                spo2 = float(base_record["spo2"])
                sbp = float(base_record["systolic_bp"])
                dbp = float(base_record["diastolic_bp"])
                rr = float(base_record["respiratory_rate"])
                temp = float(base_record["temperature"])
            else:
                hr = float(np.clip(np.random.normal(74.0, 3.0), 55.0, 100.0))
                spo2 = float(np.clip(np.random.normal(98.4, 0.6), 95.0, 100.0))
                sbp = float(np.clip(np.random.normal(120.0, 4.0), 105.0, 135.0))
                dbp = float(np.clip(np.random.normal(78.0, 3.0), 65.0, 85.0))
                rr = float(np.clip(np.random.normal(16.0, 1.2), 12.0, 20.0))
                temp = float(np.clip(np.random.normal(37.0, 0.15), 36.6, 37.4))

            reading = {
                "patient_id": patient_id,
                "timestamp": round(now, 2),
                "heart_rate": round(hr, 1),
                "spo2": round(spo2, 1),
                "systolic_bp": round(sbp, 1),
                "diastolic_bp": round(dbp, 1),
                "respiratory_rate": round(rr, 1),
                "temperature": round(temp, 2),
            }
        else:
            anomaly_kind = random.choice([
                "severe_hypoxia",
                "severe_tachycardia",
                "hypertensive_spike",
                "hypotensive_shock",
                "severe_hyperthermia",
            ])

            if anomaly_kind == "severe_hypoxia":
                hr = random.uniform(115.0, 138.0)
                spo2 = random.uniform(72.0, 87.5)
                sbp = 120.0
                dbp = 78.0
                rr = random.uniform(26.0, 36.0)
                temp = 37.0
            elif anomaly_kind == "severe_tachycardia":
                hr = random.uniform(142.0, 185.0)
                spo2 = random.uniform(92.0, 96.0)
                sbp = random.uniform(130.0, 145.0)
                dbp = random.uniform(85.0, 95.0)
                rr = random.uniform(22.0, 28.0)
                temp = 37.0
            elif anomaly_kind == "hypertensive_spike":
                hr = random.uniform(105.0, 125.0)
                spo2 = 96.5
                sbp = random.uniform(182.0, 225.0)
                dbp = random.uniform(110.0, 132.0)
                rr = 20.0
                temp = 37.0
            elif anomaly_kind == "hypotensive_shock":
                hr = random.uniform(125.0, 150.0)
                spo2 = random.uniform(89.0, 94.0)
                sbp = random.uniform(62.0, 82.0)
                dbp = random.uniform(38.0, 50.0)
                rr = random.uniform(24.0, 32.0)
                temp = random.uniform(38.5, 39.8)
            else:  # severe_hyperthermia
                hr = random.uniform(120.0, 140.0)
                spo2 = 94.0
                sbp = 115.0
                dbp = 75.0
                rr = 22.0
                temp = random.uniform(39.3, 40.8)

            reading = {
                "patient_id": patient_id,
                "timestamp": round(now, 2),
                "heart_rate": round(hr, 1),
                "spo2": round(spo2, 1),
                "systolic_bp": round(sbp, 1),
                "diastolic_bp": round(dbp, 1),
                "respiratory_rate": round(rr, 1),
                "temperature": round(temp, 2),
            }

        yield reading
        await asyncio.sleep(interval)
