"""
Anomaly detection model combining Isolation Forest with deterministic medical rule guardrails.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from src.preprocessor import VitalSignsPreprocessor

DEFAULT_MODEL_PATH = Path("models/anomaly_detector.joblib")


class PatientAnomalyDetector:
    """Hybrid anomaly detection model pairing an unsupervised Isolation Forest with clinical guardrails."""

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 150,
        random_state: int = 42,
        preprocessor: Optional[VitalSignsPreprocessor] = None,
    ) -> None:
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.preprocessor = preprocessor

        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray) -> PatientAnomalyDetector:
        """Fit Isolation Forest on scaled training features.

        Args:
            X: 2D numpy array of shape (n_samples, n_features).

        Returns:
            Self instance.
        """
        if len(X.shape) != 2:
            raise ValueError(f"Expected 2D array, got shape {X.shape}")

        self.model.fit(X)
        self.is_fitted = True
        return self

    def _check_clinical_guardrails(self, vitals: Dict[str, Any]) -> List[str]:
        """Evaluate deterministic clinical rule guardrails against physiological safety bounds.

        Trigger immediate CRITICAL alerts if:
        - heart_rate < 45 or > 130 bpm
        - spo2 < 90%
        - systolic_bp > 170 or < 85 mmHg
        - temperature > 39.0°C (102.2°F)
        """
        reasons: List[str] = []

        try:
            hr = float(vitals.get("heart_rate", 75.0))
            spo2 = float(vitals.get("spo2", 98.0))
            sbp = float(vitals.get("systolic_bp", 120.0))
            temp = float(vitals.get("temperature", 37.0))
        except (ValueError, TypeError):
            return ["Invalid vital signs data format"]

        # Heart Rate Rules
        if hr < 45.0:
            reasons.append(f"Severe bradycardia: Heart Rate {hr:.1f} bpm < 45 bpm")
        elif hr > 130.0:
            reasons.append(f"Severe tachycardia: Heart Rate {hr:.1f} bpm > 130 bpm")

        # Oxygen Saturation Rule
        if spo2 < 90.0:
            reasons.append(f"Critical hypoxia: SpO2 {spo2:.1f}% < 90.0%")

        # Blood Pressure Rules
        if sbp > 170.0:
            reasons.append(f"Severe hypertensive crisis: Systolic BP {sbp:.1f} mmHg > 170.0 mmHg")
        elif sbp < 85.0:
            reasons.append(f"Severe hypotensive shock: Systolic BP {sbp:.1f} mmHg < 85.0 mmHg")

        # Core Body Temperature Rule
        if temp > 39.0:
            reasons.append(f"Severe hyperthermia: Core Temperature {temp:.2f}°C > 39.0°C")

        return reasons

    def predict(self, raw_vitals: Dict[str, Any]) -> Dict[str, Any]:
        """Perform combined anomaly inference using ML model and rule guardrails.

        Args:
            raw_vitals: Dictionary of vital sign readings.

        Returns:
            Dictionary with:
            - is_anomaly: bool
            - anomaly_score: float (normalized 0.0 to 1.0)
            - severity: "NORMAL", "WARNING", or "CRITICAL"
            - flagged_reasons: list of triggered rules or ML anomaly markers.
        """
        flagged_reasons: List[str] = []

        # 1. Deterministic Rule Guardrails
        critical_violations = self._check_clinical_guardrails(raw_vitals)
        has_critical_guardrail = len(critical_violations) > 0
        if has_critical_guardrail:
            flagged_reasons.extend(critical_violations)

        # 2. Machine Learning Inference (Isolation Forest)
        ml_score = 0.0
        ml_is_anomaly = False

        if self.is_fitted:
            # Resolve preprocessor if available
            preprocessor = self.preprocessor
            if preprocessor is None:
                scaler_path = Path("models/scaler.joblib")
                if scaler_path.exists():
                    preprocessor = VitalSignsPreprocessor.load(scaler_path)
                    self.preprocessor = preprocessor

            if preprocessor is not None and preprocessor.is_fitted:
                try:
                    X_scaled = preprocessor.transform(raw_vitals)
                    df_val = float(self.model.decision_function(X_scaled)[0])
                    ml_pred = int(self.model.predict(X_scaled)[0])

                    # Sigmoidal mapping centered on decision threshold (df_val = 0.0)
                    # When df_val > 0 (inlier), score < 0.50 (normal)
                    # When df_val < 0 (outlier), score > 0.50 (anomaly risk)
                    ml_score = float(1.0 / (1.0 + np.exp(10.0 * df_val)))
                    ml_is_anomaly = bool(ml_pred == -1 or ml_score >= 0.55)

                    if ml_is_anomaly:
                        flagged_reasons.append(
                            f"Isolation Forest flagged multivariate outlier (risk score: {ml_score:.2f})"
                        )
                except Exception as exc:
                    flagged_reasons.append(f"ML inference error: {exc}")

        # 3. Aggregate Decision & Severity
        if has_critical_guardrail:
            severity = "CRITICAL"
            is_anomaly = True
            normalized_score = max(ml_score, 0.95)
        elif ml_is_anomaly:
            severity = "CRITICAL" if ml_score >= 0.85 else "WARNING"
            is_anomaly = True
            normalized_score = ml_score
        elif ml_score >= 0.40:
            severity = "WARNING"
            is_anomaly = False
            normalized_score = ml_score
        else:
            severity = "NORMAL"
            is_anomaly = False
            normalized_score = ml_score

        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": round(float(normalized_score), 4),
            "severity": severity,
            "flagged_reasons": flagged_reasons,
        }

    def save(self, filepath: Union[str, Path] = DEFAULT_MODEL_PATH) -> None:
        """Persist model artifact using joblib."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, filepath: Union[str, Path] = DEFAULT_MODEL_PATH) -> PatientAnomalyDetector:
        """Load persisted detector model."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found at {path}")
        return joblib.load(path)
