"""Reproducible training entry point for the UPI fraud-detection model.

    DATA -> PREPROCESSING -> FEATURE ENGINEERING -> TRAINING -> VALIDATION
    -> EVALUATION -> MODEL ARTIFACT

Usage (from the repo root, with dependencies installed):

    python ml/src/train.py

Produces:
    ml/artifacts/models/upi_fraud_model.pkl   (the fitted pipeline)
    ml/artifacts/metrics/metrics.json         (evaluation metrics + run metadata)

This replaces ad hoc, non-reproducible notebook training with a single
deterministic script (fixed random_state throughout) that can be re-run to
regenerate the exact artifact this codebase serves.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.src.data.load import load_features_and_target
from ml.src.evaluation.metrics import compute_metrics
from ml.src.features.schema import FEATURE_COLUMNS
from ml.src.training.pipeline import N_ESTIMATORS, RANDOM_STATE, build_pipeline
from ml.src.utils.paths import METRICS_PATH, MODEL_PATH, RAW_DATASET_PATH, REPO_ROOT

TEST_SIZE = 0.2


def train():
    # DATA
    X, y = load_features_and_target(RAW_DATASET_PATH)

    # split before any fitting happens, so the test set is never seen by
    # the preprocessor or the classifier (avoids train/test leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # PREPROCESSING + FEATURE ENGINEERING + TRAINING
    # (bundled into one sklearn Pipeline so preprocessing always travels
    # with the model artifact — no separate encoder to keep in sync)
    pipeline = build_pipeline(random_state=RANDOM_STATE, n_estimators=N_ESTIMATORS)
    pipeline.fit(X_train, y_train)

    # VALIDATION / EVALUATION
    y_pred = pipeline.predict(X_test)
    classes = list(pipeline.named_steps['classifier'].classes_)
    y_proba_positive = pipeline.predict_proba(X_test)[:, classes.index('Y')]
    metrics = compute_metrics(y_test, y_pred, y_proba_positive)

    # MODEL ARTIFACT
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    run_record = {
        'model_name': 'RandomForestClassifier',
        'dataset_path': str(RAW_DATASET_PATH.relative_to(REPO_ROOT)),
        'dataset_rows': int(len(X)),
        'train_rows': int(len(X_train)),
        'test_rows': int(len(X_test)),
        'test_size': TEST_SIZE,
        'random_state': RANDOM_STATE,
        'n_estimators': N_ESTIMATORS,
        'sklearn_version': sklearn.__version__,
        'trained_at_utc': datetime.now(timezone.utc).isoformat(),
        'feature_columns': FEATURE_COLUMNS,
        'metrics': metrics,
    }
    with open(METRICS_PATH, 'w') as f:
        json.dump(run_record, f, indent=2)

    print(json.dumps(run_record, indent=2))
    return run_record


if __name__ == '__main__':
    train()
