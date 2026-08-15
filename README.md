# Smart UPI Fraud Detection

A Django web application where a registered user submits a UPI transaction's
attributes and gets a real-time fraud prediction — with confidence score and
top contributing factors — from a reproducibly-trained scikit-learn model.
Predictions are persisted per-user in the database.

## 1. Problem statement

UPI (India's real-time payments rail) fraud is typically flagged after the
fact, from patterns like abnormal transaction amounts, repeated declines,
foreign transactions from high-risk countries, and chargeback history. This
project takes 9 such transaction-level attributes and classifies a
transaction as fraudulent or legitimate, exposing that model through a
usable web app rather than leaving it in a notebook.

## 2. Solution

A trained `RandomForestClassifier` (wrapped in a scikit-learn `Pipeline`
with its preprocessing) is served behind a Django view. A user logs in,
fills in a transaction's attributes, and gets back a label, a fraud
probability, and the model's top global feature importances. Every
prediction is saved against that user's account and viewable in a personal
history page.

## 3. System architecture

```
Transaction form (Django template)
        |
        v
Django view (users/views/prediction.py)  -- validates input
        |
        v
prediction_service.py  -- Django-side adapter
        |
        v
ml/src/inference/predictor.py (FraudPredictor)
        |
        v
ml/artifacts/models/upi_fraud_model.pkl
  (ColumnTransformer[StandardScaler + OneHotEncoder] -> RandomForestClassifier)
        |
        v
label + fraud_probability + top_features
        |
        v
UserPredictModel row (owned by request.user) --> SQLite
        |
        v
/database/ -- per-user prediction history
```

See `docs/architecture.md` for the full walkthrough and design rationale,
and `docs/AUDIT.md` for the complete 23-category engineering audit this
project was built against.

## 4. ML pipeline

`ml/src/train.py` is the single reproducible entry point:

```
DATA (data/raw/UPI_FRAUD.csv)
  -> drop non-predictive columns (Merchant_id, TransactionDate)
  -> stratified 80/20 train/test split (random_state=42)
  -> PREPROCESSING + FEATURE ENGINEERING (ColumnTransformer, fit on train only)
  -> TRAINING (RandomForestClassifier, random_state=42)
  -> VALIDATION / EVALUATION (held-out test set)
  -> MODEL ARTIFACT (ml/artifacts/models/upi_fraud_model.pkl)
                    + ml/artifacts/metrics/metrics.json
```

The original notebooks (`ml/notebooks/M1`-`M6`) explored preprocessing and
compared Naive Bayes/Decision Tree/Random Forest, but none of them, as
written, produced the exact pipeline this app used to serve. That pipeline's
architecture was recovered by directly introspecting the previously-shipped
`.pkl` artifact and is now reproduced verbatim in
`ml/src/training/pipeline.py` — see `ml/README.md` for the full story.

## 5. Feature engineering

9 raw transaction fields, split into two groups (`ml/src/features/schema.py`
is the single source of truth both training and inference import from):

| Type | Features |
|---|---|
| Numeric (scaled with `StandardScaler`) | `AverageAmountTransactionDay`, `TransactionAmount`, `TotalNumberOfDeclinesDay`, `DailyChargebackAvgAmt`, `Six_MonthAvgChbkAmt`, `Six_MonthChbkFreq` |
| Categorical, Y/N (`OneHotEncoder(drop='first')`) | `Is_declined`, `isForeignTransaction`, `isHighRiskCountry` |

Target: `isFradulent` (`Y`/`N`, no encoding — the classifier is trained
directly on the raw labels).

## 6. Model selection

`RandomForestClassifier(n_estimators=200, random_state=42)` — this was the
architecture already being served in production (see Section 4), and the
original notebooks' own Naive Bayes/Decision Tree/Random Forest comparison
had already pointed the same way. Preprocessing and the classifier are
bundled into one `sklearn.pipeline.Pipeline`, so there's no separate encoder
to keep in sync with the model at inference time.

## 7. Evaluation metrics

Fraud detection is an imbalanced classification problem (in this dataset,
the test split is ~90 fraud out of 615 transactions, ~14.6%), so accuracy
alone is not a reliable signal — a model that always predicts "not fraud"
would still score ~85% accuracy while catching zero fraud. **Recall** (of
actual fraud cases, how many did we catch) and **precision** (of the cases
we flagged as fraud, how many really were) are the metrics that matter here,
and they trade off against each other: a lower classification threshold
catches more fraud (higher recall) at the cost of more false alarms (lower
precision). ROC-AUC and PR-AUC summarize that trade-off across all
thresholds.

Measured on a held-out 20% stratified test split (`ml/artifacts/metrics/metrics.json`,
reproducible via `python ml/src/train.py`):

| Metric | Value |
|---|---|
| Accuracy | 0.989 |
| Precision | 0.966 |
| Recall | 0.956 |
| F1 | 0.961 |
| ROC-AUC | 0.995 |
| PR-AUC (average precision) | 0.985 |

Confusion matrix (test set, n=615): TN=522, FP=3, FN=4, TP=86.

These numbers are only as good as the dataset (3,075 rows, a Kaggle-style
synthetic/curated UPI fraud dataset) — see Section 18, Limitations.

## 8. Explainability

Each prediction shows a fraud probability (`predict_proba` on the fitted
pipeline) and the model's top-3 **global** feature importances
(`RandomForestClassifier.feature_importances_`, mapped back to human-readable
column names). This is a model-level importance ranking, not a per-prediction
explanation — no SHAP/LIME is implemented, and the result page says so.

## 9. Inference architecture

`ml/src/inference/predictor.py` (`FraudPredictor`) loads the artifact once
and never retrains. It validates input against the feature schema
(`validate_input`, raising `InvalidInputError` with a specific message on
missing/malformed fields) before ever calling `.predict()`. Django's
`backend/users/services/prediction_service.py` is a thin adapter that
instantiates one `FraudPredictor` at import time (module-level singleton —
loaded once per worker process, not per request) and is the only place the
Django app touches the model.

## 10. Django integration

Single Django app (`users`) — deliberately not split into separate
`accounts`/`fraud_detection` apps, to avoid rewriting migration/
`django_content_type` history against the existing local database (see
`docs/architecture.md`). Internally split into `views/{auth,prediction,reports,pages}.py`
and `services/prediction_service.py` for separation of concerns. Both
`prediction_view` and `analyze_upi`/`model_db_view` now require
`@login_required` (previously they didn't).

## 11. Database flow

`UserPredictModel` (extended, not replaced) stores every prediction:
`user` (FK, owns the row), `created_at`, the 9 input fields, `isFradulent`
(the predicted label), and `fraud_probability`. `/database/` queries
`UserPredictModel.objects.filter(user=request.user)` — **this is the fix for
a real data-exposure bug** found during the audit, where every logged-in
user could previously see every other user's prediction history via an
in-memory list shared across the whole process. 27 pre-existing rows from
before user-tracking existed were left in place with `user=NULL` rather than
deleted or reassigned; they're excluded from every user's view by the same
filter.

## 12. Installation

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

pip install -r requirements.txt

cp .env.example backend/.env   # then set SECRET_KEY (anything random for local dev)

cd backend
python manage.py migrate
python manage.py createsuperuser
```

## 13. Training

```bash
# from the repo root
python ml/src/train.py
```

Deterministically retrains and overwrites
`ml/artifacts/models/upi_fraud_model.pkl` + `ml/artifacts/metrics/metrics.json`.
The Django app picks up the new artifact next time it starts.

## 14. Evaluation

Evaluation runs as part of `ml/src/train.py` (held-out test split, computed
once per training run) — there is no separate evaluate-only script, since
this project has one dataset and one held-out split, not a
train/deploy/monitor loop with new data to evaluate against yet.
`ml/artifacts/metrics/metrics.json` is machine-readable and versioned in git,
so every commit's metrics are traceable.

## 15. Running the application

```bash
cd backend
python manage.py runserver
```

Open http://localhost:8000/, register or log in, then use **Model** in the
nav for the prediction form.

## 16. Deployment

```bash
docker build -t smart-upi-fraud-detect .
docker run -p 8000:8000 --env-file backend/.env smart-upi-fraud-detect
```

The image installs `requirements.txt`, runs `collectstatic` at build time
(served via `whitenoise`, no separate nginx needed), and at container start
runs migrations then `gunicorn config.wsgi:application`. `GET /healthz/`
returns `{"status": "ok"}` for load balancer / orchestrator health checks.
`DEBUG` defaults to `True` for local development; set `DEBUG=False` (and a
real `SECRET_KEY`, `ALLOWED_HOSTS`) for any non-local deployment — with
`DEBUG=False`, `SECRET_KEY` unset now raises `ImproperlyConfigured` at
startup instead of silently booting insecurely.

## 17. Testing

```bash
cd backend
python manage.py test users
```

Covers: the model artifact loading, deterministic prediction on fixed
input, input-validation rejecting malformed data, the full authenticated
prediction flow persisting to the database, and — as a regression test for
the fix in Section 11 — that one user cannot see another user's prediction
history and that anonymous requests are redirected to login.

CI (`.github/workflows/ci.yml`) runs `manage.py check` and the test suite on
every push.

## 18. Limitations

- The dataset is small (3,075 rows) and its provenance (real vs. synthetic
  UPI transactions) isn't documented — the metrics in Section 7 describe
  this dataset, not real-world production fraud rates.
- Inference is synchronous, in-process, single-instance — there's no queue,
  batch scoring, or horizontal scaling story.
- Explainability is a global feature-importance ranking, not a
  per-prediction explanation (no SHAP/LIME).
- No experiment tracking, model registry, drift detection, or automated
  retraining — `ml/src/train.py` is a manually-run script.
- No CD (CI runs tests only; there's no automated deploy step).
- SQLite is fine for this scale/demo; it isn't a production-grade database.

## 19. Future improvements

See `docs/AUDIT.md` for the full prioritized roadmap (Phase A/B/C). The
highest-leverage remaining items: experiment tracking + a model registry,
richer per-prediction explainability, structured logging, and a proper
CD pipeline to a live deployment.

## 20. Resume highlights

- Reverse-engineered an undocumented production model artifact (exact
  `ColumnTransformer`/classifier architecture recovered via introspection)
  and rebuilt it as a reproducible, version-pinned training pipeline —
  fixing a real, observed scikit-learn version-mismatch bug in the process.
- Found and fixed a live cross-user data-exposure bug (any authenticated
  user could read every other user's prediction history) via a regression
  test that still guards it.
- Shipped honest, measured evaluation metrics (precision/recall/F1/ROC-AUC/
  PR-AUC + confusion matrix) for an imbalanced classification problem,
  persisted as a versioned artifact — no invented benchmark numbers.
- End-to-end ownership: data pipeline, model training, inference service,
  Django integration, database design, tests, Docker/CI — one person,
  one repository.

## Repository layout

```
Smart-upi-fraud-detect/
├── backend/                     # Django project
│   ├── manage.py
│   ├── config/                    # settings/urls/wsgi/asgi
│   ├── users/
│   │   ├── views/{auth,prediction,reports,pages}.py
│   │   ├── services/prediction_service.py
│   │   ├── tests/                  # model, evaluation, and view tests
│   │   ├── models.py, forms.py, admin.py, signals.py, migrations/
│   │   ├── static/, templates/
│   ├── media/, db.sqlite3           # local dev data (gitignored going forward)
│   └── docker-entrypoint.sh
├── ml/
│   ├── src/{data,features,training,evaluation,inference,utils}/, train.py
│   ├── notebooks/                    # original EDA/training notebooks
│   ├── artifacts/{models,metrics}/
│   └── README.md
├── data/raw/UPI_FRAUD.csv
├── docs/{architecture.md, AUDIT.md, thesis/}
├── .github/workflows/ci.yml
├── Dockerfile, .dockerignore
├── requirements.txt, .env.example, .gitignore
```

## Notes

- `backend/db.sqlite3` and `backend/media/` already exist locally (test data
  from development) and are left in place; they're gitignored for any
  future commits, not deleted.
