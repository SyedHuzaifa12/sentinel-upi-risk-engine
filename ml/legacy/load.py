"""Dataset loading for the UPI fraud model.

Every notebook (M1-M6) repeated the same ``del df['Merchant_id']``/
``del df['TransactionDate']`` cleanup inline. This module is the one place
that logic now lives.
"""
import pandas as pd

from ml.legacy.schema import DROP_COLUMNS, FEATURE_COLUMNS, TARGET_COLUMN
from ml.src.utils.paths import RAW_DATASET_PATH


def load_dataset(csv_path=RAW_DATASET_PATH):
    """Load the raw CSV and drop the non-predictive identifier/date columns.

    Returns the full cleaned DataFrame (features + target column).
    """
    df = pd.read_csv(csv_path)
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    return df


def load_features_and_target(csv_path=RAW_DATASET_PATH):
    """Load the dataset and split it into (X, y) ready for training."""
    df = load_dataset(csv_path)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y
