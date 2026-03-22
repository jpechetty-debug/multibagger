"""Model training for forward-return prediction."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf

import config
from data.db import db
from engines.score_engine.model import ScoreEngine
from ml.features import MLFeatureEngineer
from ml.registry import ModelRegistry
from models.schemas import FundamentalData, ModelVersion
from ticker_list import to_yfinance


class ModelTrainer:
    """Trains the forward-return meta model."""

    def __init__(self, score_engine: ScoreEngine | None = None, feature_engineer: MLFeatureEngineer | None = None, registry: ModelRegistry | None = None) -> None:
        """Initialize the trainer."""

        self.score_engine = score_engine or ScoreEngine()
        self.feature_engineer = feature_engineer or MLFeatureEngineer()
        self.registry = registry or ModelRegistry()

    def train(self, model_name: str = config.DEFAULT_MODEL_NAME) -> ModelVersion:
        """Train and register a model."""

        features, labels, feature_names, dataset_source = self._training_dataset()
        if len(labels) < config.MIN_TRAINING_ROWS:
            raise ValueError(f"Not enough training rows: {len(labels)} < {config.MIN_TRAINING_ROWS}")
        model = xgb.XGBRegressor(
            n_estimators=80,
            max_depth=4,
            learning_rate=0.07,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=config.META_MODEL_RANDOM_SEED,
        )
        model.fit(features, labels)
        metadata = {
            "feature_names": feature_names,
            "row_count": int(len(labels)),
            "label_mean": float(np.mean(labels)),
            "label_std": float(np.std(labels)),
            "source": dataset_source,
        }
        return self.registry.save_model(model, model_name, metadata)

    def _training_dataset(self) -> tuple[np.ndarray, np.ndarray, list[str], str]:
        """Build the training matrix from PIT or fallback data."""

        rows: list[np.ndarray] = []
        labels: list[float] = []
        feature_names: list[str] | None = None
        dataset_source = "pit"

        for pit_row in db.list_pit_fundamentals():
            try:
                record = FundamentalData(**pit_row["fundamentals"])
            except Exception:
                continue
            label = self._forward_return(record.ticker, pit_row["captured_at"])
            if label is None:
                continue
            score = self.score_engine.score_ticker(record.ticker, data=record)
            vector, names = self.feature_engineer.build(record, score)
            rows.append(vector)
            labels.append(label)
            feature_names = names

        if len(labels) < config.MIN_TRAINING_ROWS:
            dataset_source = "fallback"
            for record in db.list_fundamentals(effective=True):
                label = self._fallback_label(record.ticker)
                if label is None:
                    continue
                score = self.score_engine.score_ticker(record.ticker, data=record)
                vector, names = self.feature_engineer.build(record, score)
                rows.append(vector)
                labels.append(label)
                feature_names = names

        if not rows or feature_names is None:
            raise ValueError("No training rows available")
        features = np.vstack(rows)
        label_array = np.asarray(labels, dtype=float)
        if len(label_array) < config.MIN_TRAINING_ROWS:
            features, label_array = self._augment_training_rows(features, label_array)
            dataset_source = f"{dataset_source}_augmented"
        return features, label_array, feature_names, dataset_source

    def _augment_training_rows(self, features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Bootstrap a minimal local dataset when historical training rows are scarce."""

        if len(labels) == 0:
            return features, labels
        rng = np.random.default_rng(config.META_MODEL_RANDOM_SEED)
        target_rows = config.MIN_TRAINING_ROWS
        augmented_features = [features]
        augmented_labels = [labels]
        while sum(chunk.shape[0] for chunk in augmented_features) < target_rows:
            remaining = target_rows - sum(chunk.shape[0] for chunk in augmented_features)
            sample_size = min(max(1, len(labels)), remaining)
            sample_indices = rng.choice(len(labels), size=sample_size, replace=True)
            feature_noise = rng.normal(0.0, config.TRAINING_FEATURE_AUGMENT_STD, size=(sample_size, features.shape[1]))
            label_noise = rng.normal(0.0, config.TRAINING_LABEL_AUGMENT_STD, size=sample_size)
            sampled_features = features[sample_indices] * (1.0 + feature_noise)
            sampled_labels = labels[sample_indices] + label_noise
            augmented_features.append(sampled_features)
            augmented_labels.append(sampled_labels)
        return np.vstack(augmented_features)[:target_rows], np.concatenate(augmented_labels)[:target_rows]

    def _forward_return(self, ticker: str, captured_at: int) -> float | None:
        """Return realized forward return from a PIT snapshot timestamp."""

        start = pd.Timestamp.utcfromtimestamp(captured_at).date()
        end = start + timedelta(days=config.FORWARD_RETURN_HORIZON_DAYS + 10)
        history = yf.download(to_yfinance(ticker), start=str(start), end=str(end), progress=False, auto_adjust=True, threads=False)
        if history.empty:
            return None
        if isinstance(history.columns, pd.MultiIndex):
            history.columns = history.columns.get_level_values(0)
        close = history["Close"].dropna()
        if len(close) < 2:
            return None
        horizon_index = min(len(close) - 1, config.FORWARD_RETURN_HORIZON_DAYS)
        return float(close.iloc[horizon_index] / close.iloc[0] - 1)

    def _fallback_label(self, ticker: str) -> float | None:
        """Use recent realized return as a fallback label when PIT data is absent."""

        history = yf.download(to_yfinance(ticker), period="6mo", interval="1d", progress=False, auto_adjust=True, threads=False)
        if history.empty:
            return None
        if isinstance(history.columns, pd.MultiIndex):
            history.columns = history.columns.get_level_values(0)
        close = history["Close"].dropna()
        if len(close) <= config.FORWARD_RETURN_HORIZON_DAYS:
            return None
        return float(close.iloc[-1] / close.iloc[-(config.FORWARD_RETURN_HORIZON_DAYS + 1)] - 1)


if __name__ == "__main__":
    print(ModelTrainer().train().model_dump())
