"""Walk-forward validation for time-series ML models."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, precision_score, recall_score, mean_absolute_error, mean_squared_error
import numpy as np
from typing import Callable, Any

def walk_forward_validate(
    df: pd.DataFrame, 
    model_fn: Callable[[], Any], 
    features: list[str], 
    target: str, 
    n_splits: int = 5, 
    gap_days: int = 2
) -> pd.DataFrame:
    """
    Expanding window walk-forward with a gap to prevent leakage.
    gap_days = days between train end and test start (avoids lookahead).
    """
    if "captured_at" in df.columns:
        df = df.sort_values("captured_at")
    
    # TimeSeriesSplit expects indices, so we'll work with the sorted dataframe
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap_days)
    results = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        
        X_train = train_df[features]
        y_train = train_df[target]
        X_test  = test_df[features]
        y_test  = test_df[target]

        model = model_fn()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        
        # Determine if it's a classifier or regressor for metrics
        is_classifier = hasattr(model, "predict_proba")
        
        metrics = {
            "fold": fold,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "train_end": train_df["captured_at"].max() if "captured_at" in train_df.columns else fold,
            "test_end": test_df["captured_at"].max() if "captured_at" in test_df.columns else fold,
        }
        
        if is_classifier:
            probs = model.predict_proba(X_test)[:, 1]
            metrics.update({
                "auc": roc_auc_score(y_test, probs),
                "precision": precision_score(y_test, probs > 0.5, zero_division=0),
                "recall": recall_score(y_test, probs > 0.5, zero_division=0),
            })
        else:
            metrics.update({
                "mae": mean_absolute_error(y_test, preds),
                "rmse": np.sqrt(mean_squared_error(y_test, preds)),
                "rank_corr": pd.Series(preds).corr(pd.Series(y_test), method="spearman")
            })
            
        results.append(metrics)

    return pd.DataFrame(results)
