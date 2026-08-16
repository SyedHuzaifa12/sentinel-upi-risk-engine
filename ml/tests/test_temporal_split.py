"""Temporal split integrity, on the real synthetic dataset. Independent of
whether training itself can complete (see PROGRESS.md's Phase 0 dominance
finding) -- this only exercises load_synthetic_events + split_by_day +
assert_temporal_integrity.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.src.data.load_synthetic import load_synthetic_events  # noqa: E402
from ml.src.training.temporal_split import (  # noqa: E402
    TEST_DAY_RANGE,
    TRAIN_DAY_RANGE,
    VAL_DAY_RANGE,
    assert_temporal_integrity,
    split_by_day,
)


@pytest.fixture(scope="module")
def synthetic_df():
    events = load_synthetic_events()
    df = pd.DataFrame({
        "timestamp": pd.to_datetime([e.timestamp for e in events], utc=True),
    })
    return df


def test_day_ranges_are_contiguous_and_non_overlapping():
    assert TRAIN_DAY_RANGE[1] + 1 == VAL_DAY_RANGE[0]
    assert VAL_DAY_RANGE[1] + 1 == TEST_DAY_RANGE[0]


def test_temporal_integrity_holds_on_real_synthetic_data(synthetic_df):
    sim_start = synthetic_df["timestamp"].min()
    train_df, val_df, test_df = split_by_day(synthetic_df, sim_start)

    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0

    assert_temporal_integrity(train_df, val_df, test_df)


def test_no_random_state_parameter_exists_on_split_by_day():
    """Regression guard for CLAUDE.md's 'NEVER random_state splits': there
    is structurally nothing to pass a random_state to."""
    import inspect

    from ml.src.training import temporal_split

    sig = inspect.signature(temporal_split.split_by_day)
    assert "random_state" not in sig.parameters
