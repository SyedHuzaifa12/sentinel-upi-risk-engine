"""Inference-only wrapper around the pre-migration (v1) RandomForest
fraud-detection pipeline.

ARCHIVED (Phase 4): this is the pre-migration path, kept for reference only
-- see ml/legacy/README.md. Nothing in service/ or the current backend/
imports this; Django now calls the FastAPI scoring service
(backend/users/services/prediction_service.py) instead.
"""
from dataclasses import dataclass, field

import joblib
import pandas as pd

from ml.legacy.schema import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, YES_NO_VALUES
from ml.src.utils.paths import MODEL_PATH


class InvalidInputError(ValueError):
    """Raised when input to the predictor does not match the expected schema."""


@dataclass
class PredictionResult:
    label: str                     # 'Y' (fraudulent) or 'N' (legitimate)
    fraud_probability: float       # model's predicted probability of the 'Y' class
    top_features: list = field(default_factory=list)  # [{"feature": str, "importance": float}, ...]


def validate_input(data: dict) -> dict:
    """Validate and coerce a raw input dict against the model's feature schema.

    Raises InvalidInputError with a human-readable message on any problem.
    Returns a new dict with numeric fields coerced to float/int.
    """
    missing = [c for c in FEATURE_COLUMNS if c not in data]
    if missing:
        raise InvalidInputError(f"Missing required field(s): {', '.join(missing)}")

    cleaned = {}
    for col in NUMERIC_FEATURES:
        try:
            cleaned[col] = float(data[col])
        except (TypeError, ValueError):
            raise InvalidInputError(f"'{col}' must be a number, got {data[col]!r}")

    for col in CATEGORICAL_FEATURES:
        value = str(data[col]).strip().upper()
        if value not in YES_NO_VALUES:
            raise InvalidInputError(f"'{col}' must be one of {sorted(YES_NO_VALUES)}, got {data[col]!r}")
        cleaned[col] = value

    return cleaned


class FraudPredictor:
    """Loads the trained pipeline once and serves predictions from it."""

    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.pipeline = joblib.load(model_path)

    def predict(self, data: dict) -> PredictionResult:
        cleaned = validate_input(data)
        df_input = pd.DataFrame([cleaned], columns=FEATURE_COLUMNS)

        label = self.pipeline.predict(df_input)[0]

        classes = list(self.pipeline.classes_) if hasattr(self.pipeline, 'classes_') else \
            list(self.pipeline.named_steps['classifier'].classes_)
        proba_row = self.pipeline.predict_proba(df_input)[0]
        fraud_probability = float(proba_row[classes.index('Y')])

        top_features = self._top_features()

        return PredictionResult(
            label=str(label),
            fraud_probability=fraud_probability,
            top_features=top_features,
        )

    def _top_features(self, top_n=3):
        """Global feature importances from the trained RandomForest, mapped
        back to human-readable raw column names. This is a model-level
        importance ranking, not a per-prediction explanation (no SHAP/LIME
        is implemented)."""
        classifier = self.pipeline.named_steps.get('classifier')
        preprocessor = self.pipeline.named_steps.get('preprocessor')
        if classifier is None or preprocessor is None or not hasattr(classifier, 'feature_importances_'):
            return []

        encoded_names = preprocessor.get_feature_names_out()
        importances = classifier.feature_importances_

        readable = []
        for name, importance in zip(encoded_names, importances):
            raw_name = name.split('__', 1)[-1]
            for cat_col in CATEGORICAL_FEATURES:
                if raw_name.startswith(cat_col):
                    raw_name = cat_col
                    break
            readable.append((raw_name, float(importance)))

        readable.sort(key=lambda pair: pair[1], reverse=True)
        return [{'feature': name, 'importance': round(importance, 4)} for name, importance in readable[:top_n]]
