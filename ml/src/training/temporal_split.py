"""Strictly temporal train/val/test split for the 90-day synthetic dataset.

CLAUDE.md: "Temporal train/val/test splits only. NEVER random_state splits."
There is no random_state parameter anywhere in this module -- structurally
nothing to pass one to. train.py never imports
sklearn.model_selection.train_test_split.
"""
TRAIN_DAY_RANGE = (0, 59)
VAL_DAY_RANGE = (60, 74)
TEST_DAY_RANGE = (75, 89)


def split_by_day(df, sim_start):
    """Splits a DataFrame with a 'timestamp' column into (train, val, test)
    by day-offset from sim_start, using the day ranges above."""
    day = (df["timestamp"] - sim_start).dt.days

    train_df = df[(day >= TRAIN_DAY_RANGE[0]) & (day <= TRAIN_DAY_RANGE[1])]
    val_df = df[(day >= VAL_DAY_RANGE[0]) & (day <= VAL_DAY_RANGE[1])]
    test_df = df[day >= TEST_DAY_RANGE[0]]

    return train_df, val_df, test_df


def assert_temporal_integrity(train_df, val_df, test_df):
    """The entire point of a temporal split: every training timestamp
    strictly precedes every validation timestamp, which strictly precedes
    every test timestamp. Raises AssertionError (fails loudly) if not."""
    train_max = train_df["timestamp"].max()
    val_min = val_df["timestamp"].min()
    val_max = val_df["timestamp"].max()
    test_min = test_df["timestamp"].min()

    assert train_max < val_min, f"train max ({train_max}) is not before val min ({val_min})"
    assert val_max < test_min, f"val max ({val_max}) is not before test min ({test_min})"
