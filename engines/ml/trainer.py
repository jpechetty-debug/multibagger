"""ML Trainer for Sovereign Engine.

FIX-3 changes
-------------
- Removed the duplicate import block (the file had two sets of `import time`,
  `import json`, etc. from a bad merge — now deduplicated).
- After a model is promoted to "Active" in the DB, ScoreEngine's class-level
  model cache is invalidated via ScoreEngine.invalidate_model_cache() and
  ModelGuard.set_baseline() is called with the new model's AUC so the guard
  has a fresh baseline to measure decay against.
- The trained SovereignEnsemble is a classifier.  predict_proba() is the
  correct inference path (FIX-3 in model.py handles this on the scoring side).

PLACEMENT: replace engines/ml/trainer.py with this file.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data.db import db
from engines.ml.calibration import calibrate_model, generate_calibration_report
from engines.ml.ensemble import SovereignEnsemble
from engines.ml.feature_logger import log_feature_importance
from engines.ml.labeler import PointInTimeLabeler
from engines.ml.validation import walk_forward_validate

logger = logging.getLogger(__name__)


class SovereignTrainer:
    """Trains robust ensembles on PIT fundamentals to predict forward return outcomes."""

    def __init__(self, target_col: str = "forward_3m_ret", threshold: float = 0.05) -> None:
        self.target_col = target_col
        self.threshold  = threshold   # forward return threshold for binary label (>5% = win)
        self.features   = [
            "market_cap", "avg_volume", "roe_5y", "roe_ttm", "sales_growth_5y",
            "eps_growth_ttm", "cfo_to_pat", "debt_equity", "peg_ratio", "pe_ratio",
            "piotroski_score", "promoter_pct", "pledge_pct", "fii_delta", "dii_delta",
        ]

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add binary target column and drop rows without a label."""
        df = df.dropna(subset=[self.target_col]).copy()
        df["target"] = (df[self.target_col] > self.threshold).astype(int)
        return df

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_and_evaluate(self, df: pd.DataFrame) -> dict[str, Any]:
        """Train ensemble model and return comprehensive metrics."""
        df = self._prepare_data(df)
        if len(df) < 60:
            raise ValueError(f"Insufficient labeled data: {len(df)} samples (need ≥60).")

        # 1. Walk-forward validation — honest out-of-sample numbers
        wf_results  = walk_forward_validate(
            df,
            lambda: SovereignEnsemble(),
            self.features,
            "target",
            n_splits=5,
            gap_days=21,    # 21 calendar days = ~15 trading days, avoids near-leakage
        )
        wf_metrics = wf_results.mean(numeric_only=True).to_dict()

        # 2. Chronological train / calibration / test split
        df          = df.sort_values("captured_at")
        split_idx   = int(len(df) * 0.70)
        train_df    = df.iloc[:split_idx]
        cal_test_df = df.iloc[split_idx:]
        cal_df, test_df = train_test_split(cal_test_df, test_size=0.50, shuffle=False)

        X_train, y_train = train_df[self.features], train_df["target"]
        X_cal,   y_cal   = cal_df[self.features],   cal_df["target"]
        X_test,  y_test  = test_df[self.features],  test_df["target"]

        # 3. Fit ensemble
        ensemble = SovereignEnsemble()
        ensemble.fit(X_train, y_train)

        # 4. Probability calibration (isotonic regression on hold-out)
        calibrated_model = calibrate_model(ensemble, X_cal, y_cal, method="isotonic")

        # 5. Final evaluation
        cal_report = generate_calibration_report(calibrated_model, X_test, y_test)

        # 6. SHAP feature importance (from XGBoost sub-model)
        try:
            shap_importance = log_feature_importance(ensemble.models["xgb"], X_train, "pending")
            importance_records = shap_importance.to_dict("records")
        except Exception as exc:
            logger.warning("SHAP importance failed: %s", exc)
            importance_records = []

        return {
            "metrics": {
                "wf_auc":          float(wf_metrics.get("auc", 0.0)),
                "wf_precision":    float(wf_metrics.get("precision", 0.0)),
                "wf_recall":       float(wf_metrics.get("recall", 0.0)),
                "brier_score":     cal_report["brier_score"],
                "calibration_gap": cal_report["calibration_gap"],
                "test_auc":        float(wf_results.iloc[-1]["auc"]) if not wf_results.empty else 0.0,
            },
            "feature_importance": importance_records,
            "calibration_data":   cal_report,
            "train_size":         len(train_df),
            "test_size":          len(test_df),
            "model_obj":          calibrated_model,
        }

    # ------------------------------------------------------------------
    # Pipeline entry point
    # ------------------------------------------------------------------

    def run_training_pipeline(self) -> str:
        """Run full ML pipeline, persist model, notify ScoreEngine and ModelGuard.

        Returns
        -------
        version_id : str
            The DB version id of the newly trained model (stage = "Staging").
            Use sovereign-cli ml-ops --promote <version_id> to activate it.
        """
        labeler = PointInTimeLabeler()
        df      = labeler.generate_labeled_dataset()
        if df.empty:
            raise ValueError(
                "Labeler returned empty dataset. "
                "Ensure the daily PIT snapshot job is enabled in the scheduler "
                "so fundamentals_pit accumulates historical records."
            )

        results      = self.train_and_evaluate(df)
        version_id   = f"ens_{int(time.time())}"
        artifact_dir = "runtime/models"
        artifact_path = f"{artifact_dir}/{version_id}.joblib"

        os.makedirs(artifact_dir, exist_ok=True)
        joblib.dump(results["model_obj"], artifact_path)
        logger.info("Saved ensemble model artifact: %s", artifact_path)

        db.initialize()
        metadata = {
            "metrics":          results["metrics"],
            "feature_importance": results["feature_importance"],
            "calibration":      results["calibration_data"],
            "train_samples":    results["train_size"],
            "test_samples":     results["test_size"],
            "threshold":        self.threshold,
            "features":         self.features,
            "target_col":       self.target_col,
        }

        with db.connection("ops") as conn:
            conn.execute(
                """
                INSERT INTO model_versions
                    (version, model_name, stage, created_at, artifact_path, metadata_json, active)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    version_id,
                    f"ensemble_{self.target_col}",
                    "Staging",
                    int(time.time()),
                    artifact_path,
                    json.dumps(metadata),
                    0,       # not active until explicitly promoted
                ),
            )

        logger.info("Model %s saved to DB (stage=Staging). Run --promote to activate.", version_id)
        return version_id

    def promote(self, version_id: str) -> None:
        """Activate a Staging model and wire it into ScoreEngine + ModelGuard.

        Calling this is equivalent to `sovereign-cli ml-ops --promote <version_id>`.
        """
        with db.connection("ops") as conn:
            # Deactivate all current active models for the same name
            conn.execute(
                "UPDATE model_versions SET active = 0, stage = 'Archived' "
                "WHERE model_name = ? AND active = 1",
                (f"ensemble_{self.target_col}",),
            )
            conn.execute(
                "UPDATE model_versions SET active = 1, stage = 'Active' WHERE version = ?",
                (version_id,),
            )

        # FIX-3: invalidate ScoreEngine cache so next score_ticker() reloads
        try:
            from engines.score_engine.model import ScoreEngine
            ScoreEngine.invalidate_model_cache()
            logger.info("ScoreEngine model cache invalidated.")
        except Exception as exc:
            logger.warning("Could not invalidate ScoreEngine cache: %s", exc)

        # FIX-1: tell ModelGuard about the new baseline AUC so decay detection resets
        try:
            from engines.ml.model_guard import ModelGuard
            with db.connection("ops") as conn:
                row = conn.execute(
                    "SELECT metadata_json FROM model_versions WHERE version = ?",
                    (version_id,),
                ).fetchone()
            meta     = json.loads(row["metadata_json"]) if row else {}
            wf_auc   = float((meta.get("metrics") or {}).get("wf_auc", 0.55))
            ModelGuard().set_baseline(model_id=version_id, baseline_auc=wf_auc)
            logger.info("ModelGuard baseline set: model=%s auc=%.3f", version_id[:8], wf_auc)
        except Exception as exc:
            logger.warning("Could not notify ModelGuard: %s", exc)

        logger.info("Model %s is now Active.", version_id)
