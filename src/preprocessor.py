"""
Feature engineering and scaling pipeline for patient vital signs telemetry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DEFAULT_SCALER_PATH = Path("models/scaler.joblib")


class VitalSignsPreprocessor:
    """Preprocesses raw vital signs records and computes physiological indices.

    Derived features:
    - pulse_pressure: systolic_bp - diastolic_bp
    - mean_arterial_pressure: diastolic_bp + (pulse_pressure / 3)
    """

    NUMERICAL_FEATURES: List[str] = [
        "heart_rate",
        "spo2",
        "systolic_bp",
        "diastolic_bp",
        "respiratory_rate",
        "temperature",
    ]

    DERIVED_FEATURES: List[str] = [
        "pulse_pressure",
        "mean_arterial_pressure",
    ]

    ALL_FEATURES: List[str] = NUMERICAL_FEATURES + DERIVED_FEATURES

    def __init__(self, scaler: Optional[StandardScaler] = None) -> None:
        """Initialize preprocessor with an optional pre-fitted StandardScaler."""
        self.scaler: StandardScaler = scaler or StandardScaler()
        self.is_fitted: bool = scaler is not None

    def _compute_derived_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive clinical indices from a DataFrame."""
        data = df.copy()

        # Ensure all required numerical features exist
        for col in self.NUMERICAL_FEATURES:
            if col not in data.columns:
                raise KeyError(f"Missing required vital column: '{col}'")

        pulse_pressure = data["systolic_bp"] - data["diastolic_bp"]
        mean_arterial_pressure = data["diastolic_bp"] + (pulse_pressure / 3.0)

        data["pulse_pressure"] = pulse_pressure
        data["mean_arterial_pressure"] = mean_arterial_pressure

        return data[self.ALL_FEATURES]

    def fit(self, df: pd.DataFrame, persist_path: Optional[Union[str, Path]] = DEFAULT_SCALER_PATH) -> VitalSignsPreprocessor:
        """Fit StandardScaler on training DataFrame and persist to disk.

        Args:
            df: Historical patient vitals DataFrame.
            persist_path: File path to save fitted scaler. Defaults to 'models/scaler.joblib'.

        Returns:
            Self instance.
        """
        features_df = self._compute_derived_df(df)
        self.scaler.fit(features_df.values)
        self.is_fitted = True

        if persist_path:
            self.save(persist_path)

        return self

    def fit_transform(
        self,
        df: pd.DataFrame,
        persist_path: Optional[Union[str, Path]] = DEFAULT_SCALER_PATH,
    ) -> np.ndarray:
        """Fit scaler on data, persist scaler, and return transformed feature matrix.

        Args:
            df: Historical patient vitals DataFrame.
            persist_path: File path to persist fitted scaler artifact.

        Returns:
            2D numpy array of shape (n_samples, n_features).
        """
        self.fit(df, persist_path=persist_path)
        features_df = self._compute_derived_df(df)
        return self.scaler.transform(features_df.values)

    def transform(self, single_record_dict: Dict[str, Any]) -> np.ndarray:
        """Transform a single patient reading dictionary for real-time inference.

        Args:
            single_record_dict: Dictionary containing raw vital values.

        Returns:
            2D numpy array of shape (1, n_features) ready for model scoring.
        """
        if not self.is_fitted:
            raise RuntimeError("VitalSignsPreprocessor must be fitted or loaded before transform.")

        # Extract numerical metrics
        try:
            hr = float(single_record_dict["heart_rate"])
            spo2 = float(single_record_dict["spo2"])
            sbp = float(single_record_dict["systolic_bp"])
            dbp = float(single_record_dict["diastolic_bp"])
            rr = float(single_record_dict["respiratory_rate"])
            temp = float(single_record_dict["temperature"])
        except KeyError as e:
            raise KeyError(f"Missing required vital sign field in record: {e}")

        # Compute derived features
        pulse_pressure = sbp - dbp
        mean_arterial_pressure = dbp + (pulse_pressure / 3.0)

        raw_vector = np.array([[hr, spo2, sbp, dbp, rr, temp, pulse_pressure, mean_arterial_pressure]])
        return self.scaler.transform(raw_vector)

    def save(self, filepath: Union[str, Path] = DEFAULT_SCALER_PATH) -> None:
        """Persist fitted scaler to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, path)

    @classmethod
    def load(cls, filepath: Union[str, Path] = DEFAULT_SCALER_PATH) -> VitalSignsPreprocessor:
        """Load fitted scaler from disk and return configured preprocessor."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Scaler artifact not found at {path}")
        scaler = joblib.load(path)
        return cls(scaler=scaler)
