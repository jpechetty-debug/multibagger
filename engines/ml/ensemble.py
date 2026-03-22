"""Ensemble modeling for Sovereign Engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
try:
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
except ImportError:
    LGBMClassifier = None
    XGBClassifier = None

class SovereignEnsemble:
    """Robust ensemble combining XGBoost, LightGBM, and Logistic Regression."""
    
    def __init__(self, xgb_params: Dict[str, Any] | None = None, lgbm_params: Dict[str, Any] | None = None):
        self.models = {
            "xgb": XGBClassifier(**(xgb_params or {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05, "random_state": 42})),
            "lgbm": LGBMClassifier(**(lgbm_params or {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05, "random_state": 42})),
            "lr": LogisticRegression(C=0.1, max_iter=1000, random_state=42),
        }
        self.weights = {"xgb": 0.50, "lgbm": 0.35, "lr": 0.15}
        self.is_fitted = False

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> None:
        """Fit all models in the ensemble."""
        for name, model in self.models.items():
            if model is not None:
                model.fit(X, y)
        self.is_fitted = True

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Weighted average of predicted probabilities."""
        if not self.is_fitted:
             raise RuntimeError("Ensemble is not fitted.")
             
        weighted_probs = np.zeros(len(X))
        total_weight = 0.0
        
        for name, model in self.models.items():
            if model is not None:
                # Some models might fail if lightgbm/xgboost aren't installed correctly
                try:
                    prob = model.predict_proba(X)[:, 1]
                    weighted_probs += self.weights[name] * prob
                    total_weight += self.weights[name]
                except Exception:
                    continue
        
        return weighted_probs / total_weight if total_weight > 0 else weighted_probs

    def individual_probas(self, X: pd.DataFrame | np.ndarray) -> Dict[str, np.ndarray]:
        """Return per-model predictions for disagreement scoring."""
        return {name: m.predict_proba(X)[:, 1] for name, m in self.models.items() if m is not None}

    def disagreement_score(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Calculate the standard deviation of predictions across models."""
        probas = self.individual_probas(X)
        if not probas:
            return np.zeros(len(X))
        
        all_probs = np.array(list(probas.values()))
        return np.std(all_probs, axis=0)
