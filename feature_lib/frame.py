"""The single implementation of "feature values -> model-ready DataFrame."

Training (ml/src/train.py) and serving (service/scoring.py) both call this --
never a local re-implementation of the column-order/dtype logic. Two copies
of this would be exactly the train/serve skew this architecture exists to
prevent: if either drifted (a forgotten categorical cast, a different
column order), scores would silently diverge between training and serving
without either side raising an error.
"""
import pandas as pd

CATEGORICAL_FEATURES = ["amount_roundness"]


def vector_to_frame(rows: list, feature_columns: list) -> pd.DataFrame:
    """`rows` is a list of feature-name -> value dicts (one dict per event;
    pass a single-element list for one-row serving calls). Returns a
    DataFrame with exactly `feature_columns` as columns, in that order,
    `amount_roundness` cast to a pandas category, and any feature missing
    from a row's dict passed through as NaN (matching pandas' own
    behavior when a key is absent) rather than raising.
    """
    selected = [{name: row.get(name) for name in feature_columns} for row in rows]
    df = pd.DataFrame(selected, columns=feature_columns)
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df
