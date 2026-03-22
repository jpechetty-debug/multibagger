"""Feature importance logging using SHAP."""

from __future__ import annotations

import pandas as pd
import numpy as np
import shap
from typing import Any
from data.db import db
import json

def log_feature_importance(model: Any, X_train: pd.DataFrame, version_id: str) -> pd.DataFrame:
    """
    Log feature importance using SHAP and model-specific gain scores.
    """
    try:
        # Use TreeExplainer for tree-based models (XGBoost, LGBM, RF)
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_train)
        
        # If classifier, shap_vals might be a list (one per class); take the positive class (usually index 1)
        if isinstance(shap_vals, list) and len(shap_vals) > 1:
            shap_vals = shap_vals[1]
            
        shap_mean = np.abs(shap_vals).mean(axis=0)
    except Exception:
        # Fallback if SHAP fails or model is not tree-based
        shap_mean = np.zeros(len(X_train.columns))

    importance = pd.DataFrame({
        "feature": X_train.columns,
        "shap_mean": shap_mean,
    })

    # Add model-specific importance if available
    if hasattr(model, "feature_importances_"):
        importance["model_importance"] = model.feature_importances_
    elif hasattr(model, "get_booster"):
        # For XGBoost
        gain_scores = model.get_booster().get_score(importance_type="gain")
        importance["model_importance"] = importance["feature"].map(gain_scores).fillna(0)

    importance = importance.sort_values("shap_mean", ascending=False)

    # Persist to database (ensure metadata is updated in model_versions)
    # We'll return the dataframe so the caller can include it in metadata_json
    return importance
