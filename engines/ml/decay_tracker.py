"""Model decay tracking and health monitoring."""

from __future__ import annotations

import numpy as np
from collections import deque
from sklearn.metrics import roc_auc_score, mean_absolute_error
from typing import Any

class ModelDecayTracker:
    """Tracks real-time performance of a model to detect degradation."""
    
    def __init__(self, baseline_metric: float, metric_type: str = "auc", decay_threshold: float = 0.05, min_samples: int = 20):
        self.baseline_metric = baseline_metric  # Metric at training time
        self.metric_type = metric_type  # "auc" or "mae"
        self.decay_threshold = decay_threshold
        self.min_samples = min_samples
        self.rolling_window = deque(maxlen=100)  # last 100 predictions/outcomes

    def record_outcome(self, predicted: float, actual: float) -> None:
        """Record a single prediction and its realized outcome."""
        self.rolling_window.append((predicted, actual))

    def current_metric(self) -> float | None:
        """Calculate the current performance metric over the rolling window."""
        if len(self.rolling_window) < self.min_samples:
            return None
            
        preds, actuals = zip(*self.rolling_window)
        if self.metric_type == "auc":
            try:
                # Need at least one sample of each class for AUC
                if len(set(actuals)) < 2:
                    return None
                return float(roc_auc_score(actuals, preds))
            except Exception:
                return None
        elif self.metric_type == "mae":
            return float(mean_absolute_error(actuals, preds))
        return None

    def is_decayed(self) -> bool:
        """Check if the model's performance has significantly degraded."""
        current = self.current_metric()
        if current is None:
            return False
            
        if self.metric_type == "auc":
            # Higher is better, so decay means baseline > current
            return (self.baseline_metric - current) > self.decay_threshold
        else:
            # Lower is better (MAE), so decay means current > baseline
            return (current - self.baseline_metric) > self.decay_threshold

    def decay_report(self) -> dict[str, Any]:
        """Generate a human-readable health report."""
        current = self.current_metric()
        delta = 0.0
        if current is not None:
             delta = self.baseline_metric - current if self.metric_type == "auc" else current - self.baseline_metric
             
        return {
            "metric_type": self.metric_type,
            "baseline": float(self.baseline_metric),
            "current": float(current) if current is not None else None,
            "decay_delta": float(delta),
            "needs_retrain": self.is_decayed(),
            "samples": len(self.rolling_window)
        }
