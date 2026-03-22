"""
engines/ml/retrain.py
---------------------
Complete retrain pipeline wired to the feature store.
Shows exactly how ML reads data — ONLY through FeatureReader.

Run via CLI:
    python sovereign-cli.py retrain --version v1.0.0
"""

from __future__ import annotations

import logging
import pickle
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, brier_score_loss
from xgboost import XGBClassifier

from data.feature_store import (
    FeatureReader,
    ModelProvenance,
    SnapshotTester,
    CURRENT_VERSION,
)
from engines.ml.model_registry import ModelRegistry

logger = logging.getLogger(__name__)
MODEL_DIR = Path("runtime/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def run_retrain(
    version: str = CURRENT_VERSION,
    lookback_days: int = 365,
    n_wf_splits: int = 8,
    wf_gap_days: int = 5,
) -> dict:
    """
    Full retrain flow with walk-forward validation, calibration,
    feature store integration, and provenance registration.

    Returns summary dict with model_id, snapshot_id, AUC, Brier score.
    """
    end_date   = date.today().isoformat()
    start_date = (date.today() - timedelta(days=lookback_days)).isoformat()

    # ── Step 1: Load via feature store — never raw data ─────────────────────
    logger.info("Loading features from store: %s → %s (version %s)", start_date, end_date, version)
    reader = FeatureReader(version=version)
    X, y, meta = reader.load_dataset(start_date, end_date, labelled_only=True)

    if X.empty:
        raise RuntimeError(f"No labelled features found for version {version} in [{start_date}, {end_date}].")

    logger.info("Loaded %d rows, %d features", len(X), X.shape[1])

    # ── Step 2: Snapshot the dataset ─────────────────────────────────────────
    snapshot_id = reader.snapshot(X, y, meta)
    snap = reader.get_snapshot(snapshot_id)
    logger.info("Dataset snapshot: %s (hash: %s…)", snapshot_id[:8], snap["dataset_hash"][:16])

    # ── Step 3: Run snapshot tests — blocks training if data is bad ──────────
    tester = SnapshotTester()
    test_results = tester.run_all(X, y, meta, snapshot_id, reader)
    logger.info("Snapshot tests: %s", {k: "PASS" if v else "FAIL" for k, v in test_results.items()})

    # ── Step 4: Impute (store allows NaN — model doesn't) ────────────────────
    X_filled = X.fillna(X.median())

    # ── Step 5: Walk-forward validation ──────────────────────────────────────
    wf_auc = _walk_forward_validate(X_filled, y, n_splits=n_wf_splits, gap=wf_gap_days)
    logger.info("Walk-forward AUC: %.4f (mean across %d folds)", wf_auc, n_wf_splits)

    # ── Step 6: Train final model on full dataset ─────────────────────────────
    model = _build_model()
    model.fit(X_filled, y.astype(int))

    # ── Step 7: Calibrate on last 20% of data ────────────────────────────────
    split_idx = int(len(X_filled) * 0.8)
    X_cal, y_cal = X_filled.iloc[split_idx:], y.iloc[split_idx:].astype(int)
    calibrated = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    calibrated.fit(X_cal, y_cal)

    cal_probs  = calibrated.predict_proba(X_cal)[:, 1]
    brier      = brier_score_loss(y_cal, cal_probs)
    logger.info("Brier score (calibration set): %.4f", brier)

    # ── Step 8: Persist and Register via Model Registry ──────────────────────
    registry = ModelRegistry()
    model_id = registry.save_model(
        model=calibrated,
        dataset_hash=snap["dataset_hash"],
        algorithm="xgboost_calibrated",
        params=_build_model().get_params(),
        auc=wf_auc,
        brier=brier,
        features_version=version,
        notes=f"Retrain run on {len(X)} rows"
    )
    logger.info("Model saved to registry — model_id: %s", model_id)

    # ── Step 9: Regression Check & Promotion ──────────────────────────────────
    check = registry.regression_check(model_id)
    logger.info("Regression check: %s", check["recommendation"])

    if not check["regression"]:
        registry.promote_model(model_id, reason="Passed regression check")
        logger.info("Model PROMOTED to production")
    else:
        logger.warning("Model NOT promoted due to regression: %s", check["delta"])

    # Register provenance (legacy support)
    prov = ModelProvenance()
    prov.register(
        snapshot_id      = snapshot_id,
        dataset_hash     = snap["dataset_hash"],
        algorithm        = "xgboost_calibrated",
        version          = version,
        walk_forward_auc = wf_auc,
        brier_score      = brier,
        artifact_path    = str(artifact_path), # We still have artifact_path from registry.save_model but it's internal now
    )

    return {
        "model_id":        model_id,
        "snapshot_id":     snapshot_id,
        "dataset_hash":    snap["dataset_hash"],
        "walk_forward_auc": wf_auc,
        "brier_score":     brier,
        "artifact_path":   str(artifact_path),
        "rows_trained":    len(X),
        "features":        list(X.columns),
    }


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------

def _walk_forward_validate(X: pd.DataFrame, y: pd.Series, n_splits: int, gap: int) -> float:
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    aucs = []

    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx].astype(int), y.iloc[test_idx].astype(int)

        if y_te.nunique() < 2:
            continue  # can't compute AUC on single-class fold

        m = _build_model()
        m.fit(X_tr, y_tr)
        preds = m.predict_proba(X_te)[:, 1]
        aucs.append(roc_auc_score(y_te, preds))

    return float(np.mean(aucs)) if aucs else 0.0


def _build_model() -> XGBClassifier:
    return XGBClassifier(
        n_estimators   = 300,
        max_depth      = 4,
        learning_rate  = 0.05,
        subsample      = 0.8,
        colsample_bytree = 0.8,
        use_label_encoder = False,
        eval_metric    = "logloss",
        random_state   = 42,
        n_jobs         = -1,
    )
