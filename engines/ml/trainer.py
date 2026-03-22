"""ML Trainer for Sovereign Engine."""

import time
import json
import os
from typing import Any
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr
import joblib

from data.db import db
from engines.ml.labeler import PointInTimeLabeler

import time
import json
import os
from typing import Any, Dict, List
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split

from data.db import db
from engines.ml.labeler import PointInTimeLabeler
from engines.ml.ensemble import SovereignEnsemble
from engines.ml.validation import walk_forward_validate
from engines.ml.feature_logger import log_feature_importance
from engines.ml.calibration import calibrate_model, generate_calibration_report

class SovereignTrainer:
    """Trains robust ensembles on PIT fundamentals to predict forward return outcomes."""
    
    def __init__(self, target_col: str = "forward_3m_ret", threshold: float = 0.05) -> None:
        self.target_col = target_col
        self.threshold = threshold  # Threshold for binary classification (e.g., > 5% return)
        self.features = [
            "market_cap", "avg_volume", "roe_5y", "roe_ttm", "sales_growth_5y", 
            "eps_growth_ttm", "cfo_to_pat", "debt_equity", "peg_ratio", "pe_ratio", 
            "piotroski_score", "promoter_pct", "pledge_pct", "fii_delta", "dii_delta"
        ]

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add binary target and drop NaNs."""
        df = df.dropna(subset=[self.target_col])
        df["target"] = (df[self.target_col] > self.threshold).astype(int)
        return df

    def train_and_evaluate(self, df: pd.DataFrame) -> dict[str, Any]:
        """Train ensemble model and return comprehensive metrics."""
        df = self._prepare_data(df)
        if len(df) < 60:
            raise ValueError(f"Insufficient labeled data ({len(df)} samples).")
            
        # 1. Walk-forward Validation (Honest performance numbers)
        wf_results = walk_forward_validate(
            df, 
            lambda: SovereignEnsemble(), 
            self.features, 
            "target", 
            n_splits=5, 
            gap_days=2
        )
        wf_metrics = wf_results.mean(numeric_only=True).to_dict()

        # 2. Main Train/Calibration Split
        # Chronological split: train on older data, calibrate/test on newer
        df = df.sort_values("captured_at")
        split_idx = int(len(df) * 0.7)
        train_df = df.iloc[:split_idx]
        cal_test_df = df.iloc[split_idx:]
        
        # Split cal_test into calibration and test sets
        cal_df, test_df = train_test_split(cal_test_df, test_size=0.5, shuffle=False)
        
        X_train, y_train = train_df[self.features], train_df["target"]
        X_cal, y_cal = cal_df[self.features], cal_df["target"]
        X_test, y_test = test_df[self.features], test_df["target"]
        
        # 3. Fit Ensemble
        ensemble = SovereignEnsemble()
        ensemble.fit(X_train, y_train)
        
        # 4. Calibration
        calibrated_model = calibrate_model(ensemble, X_cal, y_cal, method="isotonic")
        
        # 5. Final Evaluation Report
        cal_report = generate_calibration_report(calibrated_model, X_test, y_test)
        
        # 6. SHAP Feature Importance
        # We'll use XGBoost from the ensemble for SHAP (as most representative)
        shap_importance = log_feature_importance(ensemble.models["xgb"], X_train, "pending")
        
        return {
            "metrics": {
                "wf_auc": float(wf_metrics.get("auc", 0.0)),
                "wf_precision": float(wf_metrics.get("precision", 0.0)),
                "wf_recall": float(wf_metrics.get("recall", 0.0)),
                "brier_score": cal_report["brier_score"],
                "calibration_gap": cal_report["calibration_gap"],
                "test_auc": float(wf_results.iloc[-1]["auc"]) if not wf_results.empty else 0.0
            },
            "feature_importance": shap_importance.to_dict("records"),
            "calibration_data": cal_report,
            "train_size": len(train_df),
            "test_size": len(test_df),
            "model_obj": calibrated_model
        }

    def run_training_pipeline(self) -> str:
        """Run full professional ML pipeline and persist results."""
        labeler = PointInTimeLabeler()
        df = labeler.generate_labeled_dataset()
        if df.empty:
            raise ValueError("Labeler returned empty dataset.")
            
        results = self.train_and_evaluate(df)
        
        db.initialize()
        version_id = f"ens_{int(time.time())}"
        artifact_path = f"runtime/models/{version_id}.joblib"
        
        os.makedirs("runtime/models", exist_ok=True)
        joblib.dump(results["model_obj"], artifact_path)
        
        with db.connection("ops") as conn:
            conn.execute(
                """
                INSERT INTO model_versions (version, model_name, stage, created_at, artifact_path, metadata_json, active)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    version_id, 
                    f"ensemble_{self.target_col}",
                    "Staging", 
                    int(time.time()), 
                    artifact_path, 
                    json.dumps({
                        "metrics": results["metrics"], 
                        "feature_importance": results["feature_importance"],
                        "calibration": results["calibration_data"],
                        "train_samples": results["train_size"],
                        "test_samples": results["test_size"],
                        "threshold": self.threshold
                    }),
                    0
                )
            )
            
        return version_id
