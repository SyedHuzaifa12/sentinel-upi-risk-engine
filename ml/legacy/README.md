# Legacy (v1) fraud-detection path

Archived here in Phase 4, when Django stopped hosting a model in-process and
became a client of the FastAPI scoring service (`service/`).

This is the **pre-migration** path: a RandomForest classifier trained on 9
hand-picked CSV-style fields (`AverageAmountTransactionDay`,
`TransactionAmount`, chargeback/foreign-transaction/high-risk-country flags,
etc.) with no VPA or transaction-history concept at all. It predates
`feature_lib` and the cold/warm LightGBM models entirely.

**Kept for reference only. Nothing in `service/` or the current `backend/`
imports anything from this directory.**

Contents:
- `inference/predictor.py` — the inference wrapper (`FraudPredictor`, `validate_input`).
- `schema.py` — the 9-field schema `predictor.py`/`pipeline.py` expect.
- `pipeline.py` — the RandomForest + preprocessing pipeline definition, recovered by
  introspecting the originally-shipped `upi_fraud_model.pkl` (the original training
  script was never in the repository).
- `load.py` — raw CSV loading for that dataset.
- `models/upi_fraud_model.pkl` — the trained artifact itself.

Not covered by CI (`pytest ml/tests` / `pytest feature_lib/tests` don't touch this
directory). A standalone regression test lives at `ml/legacy/test_predictor.py`
and can be run manually (`pytest ml/legacy/test_predictor.py`) if this code is
ever revisited, but it isn't part of the ongoing test suite.
