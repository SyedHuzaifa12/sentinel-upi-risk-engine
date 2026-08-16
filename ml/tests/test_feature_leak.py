"""Feature-level leak test: no single feature_lib FEATURE (not just a raw
event column) may separate fraud from legit almost perfectly on its own.

This is the check that should have caught the days_since_payer_last_paid_payee
bug (65.5% of gain in the warm model) before training ever ran -- Phase 0's
existing leak test (ml/tests/test_generator.py) only covers raw event
columns, which never included this feature_lib-derived one. See
PROGRESS.md "Bugs found and fixed."
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from feature_lib.compute import compute_features_batch  # noqa: E402
from feature_lib.registry import REGISTRY  # noqa: E402
from feature_lib.store.in_memory import InMemoryHistoryStore  # noqa: E402
from ml.src.data.load_synthetic import load_synthetic_events  # noqa: E402
from ml.src.training.temporal_split import split_by_day  # noqa: E402

MAX_SINGLE_FEATURE_AUC = 0.85


def _single_feature_auc(series: pd.Series, y: np.ndarray) -> float:
    if series.dtype.name == 'category' or series.dtype == object:
        # Categorical: score each row by that category's empirical fraud
        # rate (the standard "what could a single categorical column alone
        # tell a naive model" proxy -- same technique as Phase 0's leak test).
        rate_by_category = pd.Series(y).groupby(series.astype(str).values).transform('mean')
        return float(roc_auc_score(y, rate_by_category))

    values = series.to_numpy(dtype=float)
    mask = ~np.isnan(values)
    if mask.sum() < 2 or len(set(y[mask])) < 2:
        return 0.5
    auc = roc_auc_score(y[mask], values[mask])
    return float(max(auc, 1 - auc))


@pytest.fixture(scope="module")
def train_feature_df():
    events = load_synthetic_events()
    store = InMemoryHistoryStore()
    vectors = compute_features_batch(events, store)

    rows = []
    for event, vector in zip(events, vectors):
        row = dict(vector.values)
        row['timestamp'] = event.timestamp
        row['label_is_fraud'] = int(event.label_is_fraud)
        rows.append(row)

    df = pd.DataFrame(rows)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    sim_start = df['timestamp'].min()
    train_df, _val_df, _test_df = split_by_day(df, sim_start)
    return train_df


def test_no_single_feature_exceeds_max_auc(train_feature_df):
    y = train_feature_df['label_is_fraud'].to_numpy()
    feature_names = [spec.name for spec in REGISTRY]

    aucs = {}
    for name in feature_names:
        aucs[name] = _single_feature_auc(train_feature_df[name], y)

    ranked = sorted(aucs.items(), key=lambda kv: kv[1], reverse=True)
    print("\nTop 10 features by single-feature AUC (training split):")
    for name, auc in ranked[:10]:
        print(f"  {name:<45} {auc:.4f}")

    worst_name, worst_auc = ranked[0]
    assert worst_auc <= MAX_SINGLE_FEATURE_AUC, (
        f"'{worst_name}' alone achieves AUC {worst_auc:.4f} on the training split, "
        f"exceeding the {MAX_SINGLE_FEATURE_AUC} threshold -- likely a leak."
    )
