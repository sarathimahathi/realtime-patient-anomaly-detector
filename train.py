"""
Model training and evaluation pipeline for Realtime Patient Anomaly Detector.
Trains on uploaded human vital signs dataset.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.data_loader import load_kaggle_dataset
from src.model import PatientAnomalyDetector
from src.preprocessor import VitalSignsPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train")

UPLOADED_CSV = Path("data/raw/human_vital_signs_dataset_2024.csv")
DEFAULT_CSV = Path("data/raw/patient_vitals.csv")
DATA_PATH = str(UPLOADED_CSV if UPLOADED_CSV.exists() else DEFAULT_CSV)

SCALER_PATH = Path("models/scaler.joblib")
MODEL_PATH = Path("models/anomaly_detector.joblib")


def run_training_pipeline() -> None:
    """Execute end-to-end data ingestion, preprocessing, training, and artifact persistence."""
    logger.info(f"Step 1: Ingesting patient vitals dataset from: {DATA_PATH}...")
    df = load_kaggle_dataset(DATA_PATH)
    logger.info(f"Loaded dataset with {len(df):,} records across {df['patient_id'].nunique():,} patient(s).")

    # Limit training sample to 50,000 for optimal tree fitting speed & variance
    train_df = df.head(50000) if len(df) > 50000 else df
    logger.info(f"Using {len(train_df):,} records for IsolationForest fitting and scaler calibration.")

    logger.info("Step 2: Preprocessing and engineering clinical features...")
    preprocessor = VitalSignsPreprocessor()
    X = preprocessor.fit_transform(train_df, persist_path=SCALER_PATH)
    logger.info(f"Transformed feature matrix shape: {X.shape}. Scaler saved to: {SCALER_PATH}")

    logger.info("Step 3: Initializing and fitting PatientAnomalyDetector (IsolationForest)...")
    detector = PatientAnomalyDetector(contamination=0.05, n_estimators=150, random_state=42, preprocessor=preprocessor)
    detector.fit(X)

    logger.info("Step 4: Evaluating model score distribution...")
    raw_scores = detector.model.score_samples(X)
    preds = detector.model.predict(X)
    n_anomalies = np.sum(preds == -1)
    anomaly_pct = (n_anomalies / len(preds)) * 100.0

    logger.info(f"Training Anomaly Detection Rate: {anomaly_pct:.2f}% ({n_anomalies:,}/{len(preds):,})")
    logger.info(f"Isolation Forest Score Quantiles: 10%={np.quantile(raw_scores, 0.10):.3f}, 50%={np.median(raw_scores):.3f}, 90%={np.quantile(raw_scores, 0.90):.3f}")

    logger.info("Step 5: Testing clinical guardrails and inference...")
    normal_reading = {
        "heart_rate": 72.0,
        "spo2": 98.5,
        "systolic_bp": 118.0,
        "diastolic_bp": 78.0,
        "respiratory_rate": 16.0,
        "temperature": 37.0,
    }
    norm_result = detector.predict(normal_reading)
    logger.info(f"Normal Vital Check: is_anomaly={norm_result['is_anomaly']}, severity={norm_result['severity']}, score={norm_result['anomaly_score']}")

    critical_reading = {
        "heart_rate": 145.0,  # > 130 bpm tachycardia
        "spo2": 86.0,         # < 90% hypoxia
        "systolic_bp": 78.0,  # < 85 mmHg shock
        "diastolic_bp": 45.0,
        "respiratory_rate": 30.0,
        "temperature": 39.4,  # > 39.0°C fever
    }
    crit_result = detector.predict(critical_reading)
    logger.info(f"Critical Vital Check: is_anomaly={crit_result['is_anomaly']}, severity={crit_result['severity']}, score={crit_result['anomaly_score']}")
    logger.info(f"Flagged Reasons: {crit_result['flagged_reasons']}")

    logger.info("Step 6: Persisting model artifact...")
    detector.save(MODEL_PATH)
    logger.info(f"Trained detector successfully saved to: {MODEL_PATH}")

    # Final verification
    assert SCALER_PATH.exists(), f"Scaler not found at {SCALER_PATH}"
    assert MODEL_PATH.exists(), f"Model not found at {MODEL_PATH}"
    logger.info("Artifact verification passed!")


if __name__ == "__main__":
    run_training_pipeline()
