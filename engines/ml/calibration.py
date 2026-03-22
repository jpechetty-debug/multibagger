"""Probability calibration for ML models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
from typing import Any, Dict

def calibrate_model(base_model: Any, X_cal: pd.DataFrame | np.ndarray, y_cal: pd.Series | np.ndarray, method: str = "isotonic") -> Any:
    """
    Calibrate model probabilities using Isotonic Regression or Platt Scaling.
    """
    # model must already be fitted
    calibrated = CalibratedClassifierCV(
        base_model,
        method=method,      # "isotonic" or "sigmoid"
        cv="prefit"         # model already fitted — just calibrate outputs
    )
    calibrated.fit(X_cal, y_cal)
    return calibrated

def generate_calibration_report(model: Any, X_test: pd.DataFrame | np.ndarray, y_test: pd.Series | np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    """
    Generate calibration statistics including Brier score and reliability data.
    """
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_test)
    
    # Clip probabilities to [0, 1] if they are not
    probs = np.clip(probs, 0, 1)
    
    fraction_pos, mean_pred = calibration_curve(y_test, probs, n_bins=n_bins)
    brier = brier_score_loss(y_test, probs)

    return {
        "brier_score": float(brier),       # 0 = perfect, 0.25 = random
        "fraction_pos": fraction_pos.tolist(),
        "mean_predicted": mean_pred.tolist(),
        "calibration_gap": float(np.abs(fraction_pos - mean_pred).mean()),
        "bin_count": n_bins
    }
