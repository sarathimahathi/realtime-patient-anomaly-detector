"""
Realtime Patient Anomaly Detector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A modular, production-ready machine learning framework for streaming patient
telemetry, real-time feature preprocessing, and clinical anomaly detection.
"""

from .data_loader import EXPECTED_COLUMNS, generate_synthetic_vitals, load_kaggle_dataset, simulate_patient_stream
from .model import PatientAnomalyDetector
from .preprocessor import VitalSignsPreprocessor

# Backward compatibility aliases
PatientVitalPreprocessor = VitalSignsPreprocessor

__all__ = [
    "load_kaggle_dataset",
    "simulate_patient_stream",
    "generate_synthetic_vitals",
    "EXPECTED_COLUMNS",
    "VitalSignsPreprocessor",
    "PatientVitalPreprocessor",
    "PatientAnomalyDetector",
]
__version__ = "1.0.0"
