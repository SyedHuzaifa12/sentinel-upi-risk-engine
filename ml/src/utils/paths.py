"""Central path constants for the ml/ package.

Kept dependency-free (no Django) so this package can be imported both by
the Django backend and by standalone scripts (train.py, tests, notebooks).
"""
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ML_ROOT.parent

DATA_DIR = REPO_ROOT / 'data' / 'raw'
RAW_DATASET_PATH = DATA_DIR / 'UPI_FRAUD.csv'

SYNTHETIC_DATA_DIR = REPO_ROOT / 'data' / 'synthetic'
SYNTHETIC_EVENTS_PATH = SYNTHETIC_DATA_DIR / 'events.jsonl'

ARTIFACTS_DIR = ML_ROOT / 'artifacts'
MODELS_DIR = ARTIFACTS_DIR / 'models'
METRICS_DIR = ARTIFACTS_DIR / 'metrics'

# Legacy RandomForest artifact (ml/src/train.py's old target, still served by
# backend/ via ml/src/inference/predictor.py). Phase 2 does not touch these.
MODEL_PATH = MODELS_DIR / 'upi_fraud_model.pkl'
METRICS_PATH = METRICS_DIR / 'metrics.json'

# Phase 2 (ml/src/train.py): cold/warm LightGBM artifacts.
COLD_MODEL_PATH = MODELS_DIR / 'cold_model.pkl'
WARM_MODEL_PATH = MODELS_DIR / 'warm_model.pkl'
COLD_CALIBRATOR_PATH = MODELS_DIR / 'cold_calibrator.pkl'
WARM_CALIBRATOR_PATH = MODELS_DIR / 'warm_calibrator.pkl'
FEATURE_REGISTRY_PATH = MODELS_DIR / 'feature_registry.json'
METRICS_V2_PATH = METRICS_DIR / 'metrics_v2.json'
RELIABILITY_COLD_PATH = METRICS_DIR / 'reliability_cold.json'
RELIABILITY_WARM_PATH = METRICS_DIR / 'reliability_warm.json'

# Phase 3 (ml/src/policy/): decision policy artifacts.
POLICY_DIR = ARTIFACTS_DIR / 'policy'
THRESHOLDS_PATH = POLICY_DIR / 'thresholds.json'
THRESHOLD_CURVE_PATH = METRICS_DIR / 'threshold_curve.json'
