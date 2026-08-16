"""feature_lib.frame.vector_to_frame is the ONLY place that turns feature
values into a model-ready DataFrame -- both ml/src/train.py (training) and
service/scoring.py (serving) call it directly rather than each keeping
their own copy of the column-order/dtype logic. These tests pin down the
contract both call sites depend on.
"""
from datetime import datetime, timezone

import pandas as pd

from feature_lib.compute import compute_features
from feature_lib.frame import vector_to_frame
from feature_lib.registry import ALL_FEATURES, COLD_FEATURES
from feature_lib.store.in_memory import InMemoryHistoryStore
from feature_lib.tests.conftest import make_event


def _feature_values():
    store = InMemoryHistoryStore()
    event = make_event("t1", datetime(2026, 1, 1, tzinfo=timezone.utc), "payer@okaxis", "payee@ybl")
    return compute_features(event, store).values


def test_column_order_matches_feature_columns_exactly():
    values = _feature_values()
    df = vector_to_frame([values], ALL_FEATURES)
    assert list(df.columns) == ALL_FEATURES


def test_amount_roundness_cast_to_category():
    values = _feature_values()
    df = vector_to_frame([values], ALL_FEATURES)
    assert isinstance(df["amount_roundness"].dtype, pd.CategoricalDtype)


def test_missing_feature_key_becomes_nan_not_a_keyerror():
    values = _feature_values()
    del values["payer_txn_count_1h"]
    df = vector_to_frame([values], ALL_FEATURES)
    assert pd.isna(df["payer_txn_count_1h"].iloc[0])


def test_training_and_serving_call_sites_produce_identical_frames():
    """Simulates the two real call sites (train.py's full ALL_FEATURES frame,
    scoring.py's cold-routed COLD_FEATURES frame) against the SAME feature
    vector -- both must come from this one function, so a COLD_FEATURES
    frame must exactly equal ALL_FEATURES frame restricted to those columns."""
    values = _feature_values()
    training_style = vector_to_frame([values], ALL_FEATURES)
    serving_style = vector_to_frame([values], COLD_FEATURES)

    pd.testing.assert_frame_equal(training_style[COLD_FEATURES], serving_style)


def test_repeated_calls_are_deterministic():
    values = _feature_values()
    first = vector_to_frame([values], ALL_FEATURES)
    second = vector_to_frame([values], ALL_FEATURES)
    pd.testing.assert_frame_equal(first, second)
