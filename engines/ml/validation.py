"""Walk-forward validation for time-series ML models.

FIX-5: Calendar-day gap instead of sample-index gap
-----------------------------------------------------
The previous implementation passed `gap=gap_days` to TimeSeriesSplit, which
interprets the parameter as a number of *samples* to skip between train end
and test start — not calendar days.  With a sparse or irregular PIT dataset
(missing weekends, quarterly fundamentals, etc.) this gap is essentially
meaningless and allows near-term lookahead leakage.

The corrected implementation:
  1. Uses TimeSeriesSplit(gap=0) to get the raw chronological fold boundaries.
  2. After splitting, it filters the test set to exclude all rows whose
     `captured_at` timestamp is within `gap_days * 86400` seconds of the
     training set's last record.
  3. Skips folds where the filtered test set is too small to score (< 5 rows).

This guarantees a true calendar-day separation regardless of dataset density.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit


def walk_forward_validate(
    df:         pd.DataFrame,
    model_fn:   Callable[[], Any],
    features:   list[str],
    target:     str,
    n_splits:   int = 5,
    gap_days:   int = 21,   # FIX-5: calendar days, not sample count
    min_test_n: int = 5,    # skip fold if filtered test set is too small
) -> pd.DataFrame:
    """Expanding-window walk-forward validation with a true calendar-day gap.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a `captured_at` Unix-timestamp column for gap filtering.
    model_fn : Callable
        Factory that returns a fresh, unfitted model for each fold.
    features : list[str]
        Feature column names.
    target : str
        Target column name.
    n_splits : int
        Number of temporal folds.
    gap_days : int
        Minimum calendar days between the last training sample and the first
        test sample.  Default 21 days (~1 month) avoids momentum leakage.
    min_test_n : int
        Minimum number of test rows (after gap filtering) needed to score a fold.

    Returns
    -------
    pd.DataFrame
        One row per fold with metrics.  Folds with insufficient post-gap
        test data are silently skipped.
    """
    if "captured_at" not in df.columns:
        raise ValueError("DataFrame must contain a `captured_at` Unix-timestamp column.")

    df      = df.sort_values("captured_at").reset_index(drop=True)
    gap_sec = gap_days * 86_400   # convert days → seconds

    # Use gap=0 — we apply the calendar gap ourselves below
    tscv    = TimeSeriesSplit(n_splits=n_splits, gap=0)
    results = []

    for fold, (train_idx, raw_test_idx) in enumerate(tscv.split(df)):
        train_df = df.iloc[train_idx]
        raw_test = df.iloc[raw_test_idx]

        # ── FIX-5: enforce calendar-day gap ──────────────────────────────
        train_end_ts = int(train_df["captured_at"].max())
        test_df      = raw_test[raw_test["captured_at"] > train_end_ts + gap_sec]

        if len(test_df) < min_test_n:
            # Not enough post-gap test data — skip this fold
            continue

        X_train = train_df[features]
        y_train = train_df[target]
        X_test  = test_df[features]
        y_test  = test_df[target]

        model = model_fn()
        model.fit(X_train, y_train)

        is_classifier = hasattr(model, "predict_proba")

        metrics: dict[str, Any] = {
            "fold":       fold,
            "train_size": len(train_df),
            "test_size":  len(test_df),
            "train_end":  train_end_ts,
            "test_start": int(test_df["captured_at"].min()),
            "gap_days_actual": (int(test_df["captured_at"].min()) - train_end_ts) / 86_400,
        }

        if is_classifier:
            probs = model.predict_proba(X_test)[:, 1]
            try:
                metrics["auc"] = float(roc_auc_score(y_test, probs))
            except ValueError:
                metrics["auc"] = float("nan")   # only one class in test fold
            metrics["precision"] = float(precision_score(y_test, probs > 0.5, zero_division=0))
            metrics["recall"]    = float(recall_score(y_test,    probs > 0.5, zero_division=0))
        else:
            preds = model.predict(X_test)
            metrics["mae"]       = float(mean_absolute_error(y_test, preds))
            metrics["rmse"]      = float(np.sqrt(mean_squared_error(y_test, preds)))
            metrics["rank_corr"] = float(
                pd.Series(preds).corr(pd.Series(y_test.values), method="spearman")
            )

        results.append(metrics)

    if not results:
        raise ValueError(
            f"No valid folds produced after applying {gap_days}-day gap. "
            "Dataset may be too small or too sparse."
        )

    return pd.DataFrame(results)
