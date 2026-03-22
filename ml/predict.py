"""Active model prediction and SHAP explanation."""

from __future__ import annotations

import shap

import config
from data.db import db
from data.fetcher import DataFetcher
from engines.score_engine.model import ScoreEngine
from ml.features import MLFeatureEngineer
from ml.registry import ModelRegistry
from models.schemas import FundamentalData, PredictionResult


class ModelPredictor:
    """Loads the active model and predicts forward outcomes with probability calibration."""

    def __init__(self, score_engine: ScoreEngine | None = None, feature_engineer: MLFeatureEngineer | None = None, registry: ModelRegistry | None = None, fetcher: DataFetcher | None = None) -> None:
        """Initialize the predictor."""

        self.score_engine = score_engine or ScoreEngine()
        self.feature_engineer = feature_engineer or MLFeatureEngineer()
        self.registry = registry or ModelRegistry()
        self.fetcher = fetcher or DataFetcher()

    def predict(self, ticker: str, data: FundamentalData | None = None) -> PredictionResult:
        """Predict forward outcome probability and compute SHAP values."""

        normalized_ticker = ticker.strip().upper()
        data = data or db.get_fundamental(normalized_ticker, effective=True) or self.fetcher.fetch(normalized_ticker)
        score = self.score_engine.score_ticker(normalized_ticker, data=data)
        features, feature_names = self.feature_engineer.build(data, score)
        model, version = self.registry.load_active(config.DEFAULT_MODEL_NAME)
        
        # Handle both legacy Regressor and new Ensemble/Calibrated model
        if hasattr(model, "predict_proba"):
            # Probabilities for classification/ensemble
            predicted_prob = float(model.predict_proba(features.reshape(1, -1))[:, 1][0])
            predicted_value = predicted_prob
        else:
            # Legacy raw returns
            predicted_value = float(model.predict(features.reshape(1, -1))[0])

        # SHAP explaining (use the underlying model or the first stage of calibration)
        # CalibratedClassifierCV wraps the base model in .base_estimator or .calibrated_classifiers_
        explainer_model = model
        if hasattr(model, "calibrated_classifiers_"):
            # Use the base ensemble or estimator for SHAP
            explainer_model = model.estimator
            
        # If it's our SovereignEnsemble, use its XGBoost component for SHAP transparency
        if hasattr(explainer_model, "models") and "xgb" in explainer_model.models:
             explainer_model = explainer_model.models["xgb"]

        try:
            explainer = shap.TreeExplainer(explainer_model)
            shap_array = explainer.shap_values(features.reshape(1, -1))
            
            # Binary classification shap_values can be a list of two arrays
            if isinstance(shap_array, list) and len(shap_array) > 1:
                shap_array = shap_array[1]
                
            if hasattr(shap_array, "tolist"):
                shap_values_list = shap_array.tolist()[0]
            else:
                shap_values_list = list(shap_array[0])
            shap_values = {name: float(value) for name, value in zip(feature_names, shap_values_list)}
        except Exception as e:
            db.log_engine_event("WARNING", "ml.predict", f"SHAP failed: {str(e)}")
            shap_values = {}

        result = PredictionResult(
            ticker=normalized_ticker,
            model_version=version.version,
            predicted_forward_return=predicted_value, # Mapping prob to this field for schema compatibility
            shap_values=shap_values,
            feature_vector={name: float(value) for name, value in zip(feature_names, features)},
            generated_at=int(__import__("time").time()),
        )
        db.log_engine_event("INFO", "ml.predict", "prediction complete", {"ticker": normalized_ticker, "score": predicted_value, "model_version": version.version})
        return result


if __name__ == "__main__":
    print(ModelPredictor().predict("RELIANCE").model_dump())
