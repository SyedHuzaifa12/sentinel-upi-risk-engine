# Architecture

## Overview

This is a monolithic, server-rendered Django application. There is no
separate frontend build — pages are Django templates (Bootstrap 5 via CDN,
plus a small set of locally-served static assets) rendered directly by the
`users` app's views. The ML component is offline-trained and integrated as a
single artifact loaded at process start.

```
┌──────────────────────────────────────────────────────────┐
│                        Browser                            │
│   Django templates (users/templates/{users,app}/*.html)   │
└───────────────────────────┬────────────────────────────────┘
                            │ HTTP
┌───────────────────────────▼────────────────────────────────┐
│                    backend/config (Django project)          │
│   settings.py / urls.py / wsgi.py / asgi.py                  │
└───────────────────────────┬────────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────────┐
│                    backend/users (Django app)                │
│                                                              │
│  views/auth.py        — register / login / logout / profile  │
│  views/prediction.py   — prediction form + in-memory history  │
│  views/reports.py       — static basic/metrics report pages    │
│  views/pages.py          — awareness page, profile list         │
│                                                              │
│  services/prediction_service.py                              │
│     joblib.load(MODEL_ARTIFACT_DIR/upi_fraud_model.pkl)       │
│     predict(df) -> label                                      │
│                                                              │
│  models.py: Profile (1:1 with auth.User), UserPredictModel     │
│  (UserPredictModel is defined but not currently written to     │
│   by any view — see docs/AUDIT.md)                              │
└───────────────────────────┬────────────────────────────────┘
                            │ loads once at import time
┌───────────────────────────▼────────────────────────────────┐
│         ml/artifacts/models/upi_fraud_model.pkl               │
│         (trained by ml/notebooks/M6-RFC.ipynb)                  │
└──────────────────────────────────────────────────────────────┘
```

## Request flow: fraud prediction

1. User submits the form at `/prediction/` (`app/model.html`).
2. `views/prediction.py::analyze_upi` reads 9 POST fields, builds a
   single-row `pandas.DataFrame`.
3. `services/prediction_service.py::predict()` calls the pre-loaded
   scikit-learn model's `.predict()` on that frame.
4. The input + prediction is appended to a module-level `user_db` list
   (process memory only, reset on server restart) and rendered back via
   `app/result.html`.
5. `/database/` (`model_db_view`) renders whatever is currently in that
   in-memory list.

## Why the app stayed a single Django app

`Profile` (auth-related) and the fraud-prediction views share the same
`users` app. Splitting them into separate Django apps (e.g. `accounts` +
`fraud_detection`) was considered but rejected for this refactor because it
would require moving model classes across app boundaries and rewriting
migration/`django_content_type` history against the existing local
`db.sqlite3` — a real risk of breaking local logins with no project-scoped
git history to fall back on. Internal separation was done instead via the
`views/` and `services/` packages, which gives the same readability/testability
benefits without touching the Django app boundary. This is flagged as a
suggested follow-up in `docs/AUDIT.md` (Repository Architecture).

## Data & ML assets

- `data/raw/UPI_FRAUD.csv` — source dataset (also read by every notebook).
- `ml/notebooks/` — exploration and training notebooks (M1 preprocessing,
  M2 EDA, M4 Naive Bayes, M5 Decision Tree, M6 Random Forest, plus a
  ydata-profiling classification report).
- `ml/artifacts/models/upi_fraud_model.pkl` — the artifact actually served
  by the backend.
- `ml/artifacts/models/UPI1.pkl` — an earlier Random Forest dump from
  `M6-RFC.ipynb`, kept for provenance only; not loaded by the app.

No training/serving parity tooling (feature schema validation, model
versioning, or automated retraining) exists yet — see `docs/AUDIT.md` for
the full assessment.
