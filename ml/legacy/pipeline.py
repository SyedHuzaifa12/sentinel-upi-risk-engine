"""Model pipeline definition.

This exact architecture (StandardScaler + OneHotEncoder via a
ColumnTransformer, feeding a RandomForestClassifier) was recovered by
introspecting the previously-served ``upi_fraud_model.pkl`` artifact
(``pipeline.named_steps``, ``ColumnTransformer.transformers``, and
``classifier.get_params()``), since the original training script that
produced it was not present in the repository. It is reproduced here
verbatim rather than guessed at.
"""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.legacy.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES

RANDOM_STATE = 42
N_ESTIMATORS = 200


def build_pipeline(random_state=RANDOM_STATE, n_estimators=N_ESTIMATORS):
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_FEATURES),
            ('cat', OneHotEncoder(drop='first'), CATEGORICAL_FEATURES),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
    )

    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier),
    ])
