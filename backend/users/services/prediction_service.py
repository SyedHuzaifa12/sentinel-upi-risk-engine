"""Fraud-prediction inference service.

Thin Django-side adapter around ml.src.inference.FraudPredictor: loads the
trained artifact once at import time (module-level singleton, same
eager-loading behavior as the original implementation) and exposes the
result in a form the views can render/persist directly.
"""
from django.conf import settings

from ml.src.inference.predictor import FraudPredictor, InvalidInputError

_predictor = FraudPredictor(model_path=settings.MODEL_ARTIFACT_DIR / 'upi_fraud_model.pkl')


def predict(data: dict):
    """Run fraud inference on a raw input dict.

    Returns a ml.src.inference.predictor.PredictionResult.
    Raises InvalidInputError if `data` doesn't match the model's schema.
    """
    return _predictor.predict(data)
