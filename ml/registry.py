"""Registry for versioned ML models."""

from __future__ import annotations

from pathlib import Path
import json
import time
import uuid

import xgboost as xgb

import config
from data.db import db
from models.schemas import ModelVersion


class ModelRegistry:
    """Saves and loads versioned XGBoost models."""

    def save_model(self, model: xgb.XGBRegressor, model_name: str, metadata: dict) -> ModelVersion:
        """Persist a model artifact and metadata."""

        config.ensure_runtime_dirs()
        version = str(uuid.uuid4())
        artifact_dir = config.MODEL_DIR / model_name
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / f"{version}.json"
        model.save_model(artifact_path)
        metadata_path = artifact_dir / f"{version}.meta.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")
        model_version = ModelVersion(
            version=version,
            model_name=model_name,
            stage=config.ACTIVE_MODEL_STAGE,
            created_at=int(time.time()),
            artifact_path=str(artifact_path),
            metadata=metadata,
        )
        db.save_model_version(model_version, active=True)
        db.log_engine_event("INFO", "ml.registry", "model saved", model_version.model_dump())
        return model_version

    def load_active(self, model_name: str) -> tuple[xgb.XGBRegressor, ModelVersion]:
        """Load the active model for a given name."""

        model_version = db.get_active_model_version(model_name)
        if model_version is None:
            raise ValueError(f"No active model found for {model_name}")
        model = xgb.XGBRegressor()
        model.load_model(model_version.artifact_path)
        return model, model_version


if __name__ == "__main__":
    print(db.get_active_model_version(config.DEFAULT_MODEL_NAME))
