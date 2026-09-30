"""
Unit tests for VitalSignsPreprocessor and hybrid PatientAnomalyDetector.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_loader import generate_synthetic_vitals
from src.model import PatientAnomalyDetector
from src.preprocessor import VitalSignsPreprocessor


@pytest.fixture
def synthetic_patient_df() -> pd.DataFrame:
    """Generate 300 normal telemetry records for testing."""
    return generate_synthetic_vitals(n_samples=300, anomaly_rate=0.0)


def test_vital_signs_preprocessor_derived_features(synthetic_patient_df):
    """Verify Pulse Pressure and MAP computations."""
    preprocessor = VitalSignsPreprocessor()
    df_feat = preprocessor._compute_derived_df(synthetic_patient_df)

    assert "pulse_pressure" in df_feat.columns
    assert "mean_arterial_pressure" in df_feat.columns

    sample = synthetic_patient_df.iloc[0]
    expected_pp = sample["systolic_bp"] - sample["diastolic_bp"]
    expected_map = sample["diastolic_bp"] + (expected_pp / 3.0)

    assert np.isclose(df_feat.iloc[0]["pulse_pressure"], expected_pp, atol=1e-3)
    assert np.isclose(df_feat.iloc[0]["mean_arterial_pressure"], expected_map, atol=1e-3)


def test_vital_signs_preprocessor_fit_transform(synthetic_patient_df):
    """Verify StandardScaler transformation for batch and single records."""
    preprocessor = VitalSignsPreprocessor()
    matrix = preprocessor.fit_transform(synthetic_patient_df, persist_path=None)

    assert matrix.shape == (len(synthetic_patient_df), len(VitalSignsPreprocessor.ALL_FEATURES))

    # Single dictionary transform
    single_record = synthetic_patient_df.iloc[0].to_dict()
    vec = preprocessor.transform(single_record)
    assert vec.shape == (1, len(VitalSignsPreprocessor.ALL_FEATURES))


def test_model_guardrails_triggers():
    """Verify deterministic clinical guardrails trigger immediate CRITICAL alerts."""
    detector = PatientAnomalyDetector()

    # Normal sample
    norm = detector.predict({
        "heart_rate": 72.0,
        "spo2": 98.0,
        "systolic_bp": 120.0,
        "diastolic_bp": 80.0,
        "respiratory_rate": 16.0,
        "temperature": 37.0,
    })
    assert norm["is_anomaly"] is False
    assert norm["severity"] == "NORMAL"

    # Hypoxia rule (SpO2 < 90%)
    hypox = detector.predict({
        "heart_rate": 80.0,
        "spo2": 87.0,
        "systolic_bp": 120.0,
        "diastolic_bp": 80.0,
        "respiratory_rate": 18.0,
        "temperature": 37.0,
    })
    assert hypox["is_anomaly"] is True
    assert hypox["severity"] == "CRITICAL"
    assert any("SpO2" in r for r in hypox["flagged_reasons"])

    # Severe tachycardia (HR > 130 bpm)
    tachy = detector.predict({
        "heart_rate": 142.0,
        "spo2": 97.0,
        "systolic_bp": 125.0,
        "diastolic_bp": 82.0,
        "respiratory_rate": 20.0,
        "temperature": 37.0,
    })
    assert tachy["is_anomaly"] is True
    assert tachy["severity"] == "CRITICAL"
    assert any("tachycardia" in r.lower() for r in tachy["flagged_reasons"])

    # Hypertensive crisis (SBP > 170 mmHg)
    hyper = detector.predict({
        "heart_rate": 78.0,
        "spo2": 97.0,
        "systolic_bp": 185.0,
        "diastolic_bp": 115.0,
        "respiratory_rate": 18.0,
        "temperature": 37.0,
    })
    assert hyper["is_anomaly"] is True
    assert hyper["severity"] == "CRITICAL"
    assert any("hypertensive" in r.lower() for r in hyper["flagged_reasons"])


def test_model_persistence(synthetic_patient_df):
    """Verify saving and loading model and scaler artifacts."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        scaler_file = Path(tmp_dir) / "scaler.joblib"
        model_file = Path(tmp_dir) / "model.joblib"

        preprocessor = VitalSignsPreprocessor()
        X = preprocessor.fit_transform(synthetic_patient_df, persist_path=scaler_file)

        detector = PatientAnomalyDetector(contamination=0.05, n_estimators=30, preprocessor=preprocessor)
        detector.fit(X)
        detector.save(model_file)

        loaded_scaler = VitalSignsPreprocessor.load(scaler_file)
        loaded_model = PatientAnomalyDetector.load(model_file)
        loaded_model.preprocessor = loaded_scaler

        sample = synthetic_patient_df.iloc[0].to_dict()
        pred1 = detector.predict(sample)
        pred2 = loaded_model.predict(sample)

        assert pred1["is_anomaly"] == pred2["is_anomaly"]
        assert pred1["severity"] == pred2["severity"]
        assert pred1["anomaly_score"] == pred2["anomaly_score"]
