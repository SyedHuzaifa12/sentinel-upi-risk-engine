# ml/

Reproducible training + inference code for the UPI fraud-detection model, plus
the exploratory notebooks that preceded it.

```
ml/
├── src/
│   ├── data/load.py            # CSV loading + the drop-columns cleanup every notebook repeated
│   ├── features/schema.py       # single source of truth for feature names/order (training + serving import this)
│   ├── training/pipeline.py      # build_pipeline(): ColumnTransformer(StandardScaler+OneHotEncoder) -> RandomForestClassifier
│   ├── evaluation/metrics.py      # precision/recall/F1/ROC-AUC/PR-AUC/confusion matrix
│   ├── inference/predictor.py      # FraudPredictor: load artifact, validate input, predict + confidence + feature importances
│   ├── utils/paths.py               # path constants
│   └── train.py                      # entry point: data -> preprocessing -> training -> evaluation -> artifact
├── notebooks/                          # original EDA/training notebooks (M1-M6, Classification Report)
├── artifacts/
│   ├── models/upi_fraud_model.pkl       # the artifact ml/src/train.py produces and the Django app serves
│   └── metrics/metrics.json              # measured evaluation metrics + run metadata from the last training run
└── README.md
```

## Why this exists

The model previously served by the Django app (`upi_fraud_model.pkl`) had no
corresponding training script anywhere in the repository — only a `.pkl`
file. Its exact architecture (a `ColumnTransformer` of `StandardScaler` +
`OneHotEncoder` feeding a `RandomForestClassifier(n_estimators=200,
random_state=42)`) was recovered by introspecting the artifact directly
(`pipeline.named_steps`, `ColumnTransformer.transformers`,
`classifier.get_params()`) and is now reproduced exactly in
`ml/src/training/pipeline.py`, callable via `ml/src/train.py` instead of
living only inside a notebook or a black-box `.pkl`.

Retraining under this reproducible script also fixed a real,
observed bug: the original artifact was pickled under scikit-learn 1.2.1
while the project's own `requirements.txt` pins 1.1.3, producing version-
mismatch warnings on every load. The retrained artifact was produced under
1.1.3 — the version actually pinned — so it loads cleanly.

## Running it

```bash
# from the repo root, with requirements.txt installed
python ml/src/train.py
```

This deterministically (fixed `random_state=42`) retrains on
`data/raw/UPI_FRAUD.csv`, evaluates on a held-out 20% stratified split, and
overwrites `ml/artifacts/models/upi_fraud_model.pkl` and
`ml/artifacts/metrics/metrics.json`. The Django app picks up the new
artifact automatically on next start (see
`backend/users/services/prediction_service.py`).

## Notebooks

`ml/notebooks/` still holds the original exploratory work (`M1`
data-prep, `M2` EDA, `M4`/`M5`/`M6` Naive Bayes/Decision Tree/Random Forest
experiments, plus a `ydata-profiling` classification report). They are
useful for exploration/history but are **not** the source of truth for the
model the app serves — `ml/src/train.py` is.
