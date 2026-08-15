"""Central path constants for the ml/ package.

Kept dependency-free (no Django) so this package can be imported both by
the Django backend and by standalone scripts (train.py, tests, notebooks).
"""
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ML_ROOT.parent

DATA_DIR = REPO_ROOT / 'data' / 'raw'
RAW_DATASET_PATH = DATA_DIR / 'UPI_FRAUD.csv'

ARTIFACTS_DIR = ML_ROOT / 'artifacts'
MODELS_DIR = ARTIFACTS_DIR / 'models'
METRICS_DIR = ARTIFACTS_DIR / 'metrics'

MODEL_PATH = MODELS_DIR / 'upi_fraud_model.pkl'
METRICS_PATH = METRICS_DIR / 'metrics.json'
