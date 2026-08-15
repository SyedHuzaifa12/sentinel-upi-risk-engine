# SmartUPIFraudDetect — Repository Report

Factual analysis of the current repository state. No code was modified to produce this report.

---

## 1. Directory tree (3 levels, excluding venv/node_modules/migrations/static)

```
.
├── .dockerignore
├── .env.example
├── Dockerfile
├── README.md
├── requirements.txt
├── report.md
├── .github/
│   └── workflows/
│       └── ci.yml
├── backend/
│   ├── .env
│   ├── db.sqlite3
│   ├── docker-entrypoint.sh
│   ├── manage.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── asgi.py
│   │   ├── settings.py
│   │   ├── urls.py
│   │   └── wsgi.py
│   ├── media/
│   │   ├── default.jpg
│   │   └── profile_images/
│   └── users/
│       ├── __init__.py
│       ├── admin.py
│       ├── apps.py
│       ├── forms.py
│       ├── models.py
│       ├── signals.py
│       ├── urls.py
│       ├── services/
│       ├── templates/
│       ├── tests/
│       └── views/
├── data/
│   └── raw/
│       └── UPI_FRAUD.csv
├── docs/
│   ├── architecture.md
│   ├── AUDIT.md
│   └── thesis/
│       ├── UPI-FINAL.docx
│       └── UPI-FINAL.pptx
└── ml/
    ├── README.md
    ├── artifacts/
    │   ├── metrics/
    │   └── models/
    ├── notebooks/
    │   ├── Classification Report.ipynb
    │   ├── M1-DATA-PREPROCESSING.ipynb
    │   ├── M2-DATA VISUALIZATION.ipynb
    │   ├── M4-NB.ipynb
    │   ├── M5-DTC.ipynb
    │   └── M6-RFC.ipynb
    └── src/
        ├── __init__.py
        ├── train.py
        ├── data/
        ├── evaluation/
        ├── features/
        ├── inference/
        ├── training/
        └── utils/
```

---

## 2. Django apps

Single Django app: **`users`** (installed as `users.apps.UserConfig`), plus the third-party `social_django` app.

### Models (`backend/users/models.py`)

**`Profile`**
| Field | Type |
|---|---|
| `user` | `OneToOneField(User, on_delete=CASCADE)` |
| `avatar` | `ImageField(default='default.jpg', upload_to='profile_images')` |
| `bio` | `TextField` |

Has a custom `save()` that resizes the avatar to max 100×100 after saving.

**`UserPredictModel`**
| Field | Type |
|---|---|
| `user` | `ForeignKey(User, on_delete=CASCADE, null=True, blank=True, related_name='predictions')` |
| `created_at` | `DateTimeField(auto_now_add=True, null=True)` |
| `AverageAmountTransactionDay` | `FloatField` |
| `TransactionAmount` | `FloatField` |
| `Is_declined` | `CharField(max_length=100)` |
| `TotalNumberOfDeclinesDay` | `IntegerField` |
| `isForeignTransaction` | `CharField(max_length=100)` |
| `isHighRiskCountry` | `CharField(max_length=100)` |
| `DailyChargebackAvgAmt` | `FloatField` |
| `Six_MonthAvgChbkAmt` | `FloatField` |
| `Six_MonthChbkFreq` | `IntegerField` |
| `isFradulent` | `CharField(max_length=100)` |
| `fraud_probability` | `FloatField(null=True, blank=True)` |

`Meta.ordering = ['-created_at']`. Has a `Prediction` property that aliases `isFradulent` (kept for template compatibility with `model_db.html`).

### Views (`backend/users/views/`, split across 4 modules)

- **`auth.py`**: `home`, `index` (login-gated), `RegisterView` (class-based), `CustomLoginView` (extends `LoginView`, adds "remember me" session-expiry logic), `ResetPasswordView`, `ChangePasswordView`, `profile`, `logout_view`.
- **`prediction.py`**: `prediction_view` (renders the form), `analyze_upi` (POST handler — validates input, calls the ML service, saves `UserPredictModel`, renders `result.html`), `model_db_view` (lists `UserPredictModel.objects.filter(user=request.user)`).
- **`reports.py`**: `Basic_report`, `Metrics_report` — both just `render()` a static template with no context.
- **`pages.py`**: `profile_list` (lists all `Profile` objects, no filtering), `awareness_page`.

### Forms (`backend/users/forms.py`)

- `RegisterForm(UserCreationForm)` — first_name, last_name, username, email, password1, password2.
- `LoginForm(AuthenticationForm)` — username, password, remember_me.
- `UpdateUserForm(ModelForm)` — username, email (on `User`).
- `UpdateProfileForm(ModelForm)` — avatar, bio (on `Profile`).
- `UserPredictDataForm(ModelForm)` — the 9 `UserPredictModel` input fields (defined but not referenced by any view; the prediction view reads `request.POST` directly instead).

### URLs

`backend/users/urls.py`:
`''`, `register/`, `profile/`, `logout_view/`, `index/`, `Basic_report/`, `Metrics_report/`, `profile_list/`, `awareness/`, `prediction/`, `analyze_upi/`, `database/`.

`backend/config/urls.py` (root): `healthz/`, `admin/`, includes `users.urls` at `''`, `login/`, `logout/`, `password-reset-confirm/<uidb64>/<token>/`, `password-reset-complete/`, `password-change/`, `oauth/` (social_django), plus static/media serving.

---

## 3. ML code

- **Training**: `ml/src/train.py`. Loads data via `ml/src/data/load.py`, builds a pipeline via `ml/src/training/pipeline.py`, does an 80/20 stratified split (`random_state=42`), fits, evaluates via `ml/src/evaluation/metrics.py`, writes the artifact and `metrics.json`. Run manually: `python ml/src/train.py`.
- **Artifact storage**: `joblib.dump(pipeline, MODEL_PATH)` where `MODEL_PATH = ml/artifacts/models/upi_fraud_model.pkl` (defined in `ml/src/utils/paths.py`). The saved object is a full sklearn `Pipeline` (preprocessing + classifier bundled together).
  - A second file, `ml/artifacts/models/UPI1.pkl`, also exists in that directory but is not referenced anywhere in the codebase (`MODEL_PATH` only ever points to `upi_fraud_model.pkl`).
- **Loading/inference**: `ml/src/inference/predictor.py` — `FraudPredictor.__init__` calls `joblib.load(model_path)` (defaults to `MODEL_PATH`, overridable). `backend/users/services/prediction_service.py` instantiates one module-level `FraudPredictor` singleton at import time (`settings.MODEL_ARTIFACT_DIR / 'upi_fraud_model.pkl'`), loaded once per process, not per request.
- **Exact feature list/order the model expects** (`ml/src/features/schema.py`, `FEATURE_COLUMNS`, matches `feature_names_in_`):
  1. `AverageAmountTransactionDay`
  2. `TransactionAmount`
  3. `Is_declined`
  4. `TotalNumberOfDeclinesDay`
  5. `isForeignTransaction`
  6. `isHighRiskCountry`
  7. `DailyChargebackAvgAmt`
  8. `Six_MonthAvgChbkAmt`
  9. `Six_MonthChbkFreq`

  Of these, `NUMERIC_FEATURES` = the 6 float/int columns; `CATEGORICAL_FEATURES` = `Is_declined`, `isForeignTransaction`, `isHighRiskCountry` (each restricted to `{'Y','N'}`). Target column: `isFradulent`. Dropped from the raw CSV before training: `Merchant_id`, `TransactionDate`.
- **Model type**: `RandomForestClassifier` (`n_estimators=200`, `random_state=42`), scikit-learn 1.1.3.
- **Metrics artifact**: `ml/artifacts/metrics/metrics.json` — written by `train.py`, contains dataset path/row counts, split config, sklearn version, timestamp, feature list, and the metrics dict (`accuracy`, `precision`, `recall`, `f1_score`, `roc_auc`, `average_precision`, `confusion_matrix`).

---

## 4. requirements.txt

```
# --- Web application (Django) ---
asgiref==3.5.2
certifi==2022.9.24
cffi==2.1.1
charset-normalizer==2.1.1
cryptography==38.0.1
defusedxml==0.7.1
Django==4.1.2
idna==3.4
oauthlib==3.2.2
Pillow==9.2.0
pycparser==2.21
PyJWT==2.6.0
python-dotenv==0.19.0
python3-openid==3.2.0
pytz==2022.5
requests==2.28.1
requests-oauthlib==1.3.1
social-auth-app-django==5.0.0
social-auth-core==4.3.0
sqlparse==0.4.3
urllib3==1.26.12

# --- Production server / static files ---
gunicorn==21.2.0
whitenoise==6.6.0

# --- ML / fraud-prediction inference ---
numpy==1.23.4
pandas==1.5.1
scikit-learn==1.1.3
joblib==1.2.0

# --- Notebooks (ml/notebooks/) ---
matplotlib==3.6.2
seaborn==0.12.1
```

No `pyproject.toml` present in the repository.

---

## 5. Dockerfile / docker-compose / CI

**`Dockerfile`** (repo root):
- Base: `python:3.11-slim`.
- Installs `build-essential`, `libjpeg62-turbo-dev`, `zlib1g-dev` (needed to build `cryptography`/`Pillow` wheels).
- `pip install -r requirements.txt`.
- Copies the whole repo in.
- Sets working dir to `/app/backend`, makes `docker-entrypoint.sh` executable, runs `collectstatic` at build time with `DEBUG=True` forced (so `collectstatic` doesn't need real secrets/env at build).
- `EXPOSE 8000`, entrypoint is `docker-entrypoint.sh`.

**`docker-compose`**: none present in the repository.

**`backend/docker-entrypoint.sh`**: runs `python manage.py migrate --noinput`, then `exec gunicorn config.wsgi:application --bind 0.0.0.0:8000 --workers 3`.

**CI** (`.github/workflows/ci.yml`): triggers on `push` and `pull_request`. Single job on `ubuntu-latest`: checkout → set up Python 3.11 → `pip install -r requirements.txt` → `python manage.py check` (with `SECRET_KEY=ci-only-secret-key`) → `python manage.py test users` (same env var). No Docker build step, no linting step, no deployment step in this workflow.

---

## 6. Tests

Location: `backend/users/tests/` — 3 files, all Django `TestCase`/`SimpleTestCase`.

- **`test_evaluation.py`** — tests `ml/src/evaluation/metrics.py`'s `compute_metrics`: asserts all required metric keys are present and in `[0,1]`, and that the confusion matrix's four cells sum to the sample count.
- **`test_predictor.py`** — tests `ml/src/inference/predictor.py`:
  - model artifact loads successfully;
  - a fixed input produces a deterministic prediction (label + probability identical across repeated calls) and a valid label (`Y`/`N`);
  - `predict()` returns at most 3 top features, each with `feature`/`importance` keys;
  - input validation rejects missing fields, non-numeric values, and invalid categorical values (e.g. `Is_declined='MAYBE'`);
  - the predictor raises `InvalidInputError` (not an uncaught exception) on garbage input.
- **`test_views.py`** — application-level, via Django's test `Client`:
  - a full prediction POST saves a row for the submitting user with a valid label and non-null probability;
  - invalid input doesn't crash — it redirects back with a "Could not analyze" message and saves no row;
  - an anonymous user hitting `/prediction/` is redirected to login;
  - **isolation regression tests**: user B cannot see user A's prediction on `/database/`, user A does see their own, and an anonymous user is redirected away from `/database/` rather than shown data.

No frontend/JS tests, no test for `/Basic_report/`, `/Metrics_report/`, `/profile_list/`, or the OAuth flows.

---

## 7. "Fraud details" / "Prevention strategies" text — source

**Fully hardcoded in the template**, `backend/users/templates/app/result.html`. It is static HTML/text baked directly into the template with no template variables, no database query, and no per-transaction generation:

- "Fraud details" section (transaction pattern anomaly, high-risk indicators, chargeback velocity, etc.) — 5 fixed bullet points, identical for every fraud verdict regardless of which fields actually triggered it.
- "Prevention strategies", "Precautions to avoid fraud", "Recommended actions", "Common causes of UPI fraud" — all fixed bullet lists, same for every user/every prediction.
- The large "Understanding UPI frauds" education section (8 fraud-type cards) is duplicated verbatim in both the fraud and non-fraud branches of the template.
- The footer's `timestamp` and `random_id` (`ref: UPI-{{ prediction }}-{{ random_id|default:"8a3f" }}`) are never passed into the render context by `analyze_upi` (which only passes `prediction`, `fraud_probability`, `top_features`) — so these always fall back to the template's literal defaults (`2025-03-15 14:32` and `8a3f`) rather than reflecting the real time or a real reference ID.

The only values that are actually dynamic/per-prediction are: the fraud/not-fraud verdict itself, `fraud_probability`, and the `top_features` list (global model feature importances, not a per-transaction explanation).

---

## 8. Settings

**DB engine**: SQLite (`django.db.backends.sqlite3`), file `BASE_DIR / 'db.sqlite3'`. No environment-variable override for the DB — engine/path are hardcoded in `settings.py`; only credentials/URLs for other services are read from env.

**Env var handling**: `python-dotenv`'s `load_dotenv()` is called at the top of `settings.py`, loading `backend/.env`. Variables read via `os.getenv`: `MODEL_ARTIFACT_DIR`, `DEBUG`, `SECRET_KEY`, `ALLOWED_HOSTS`, `SECURE_SSL_REDIRECT`, `SECURE_HSTS_SECONDS`, `GITHUB_KEY`, `GITHUB_SECRET`, `GOOGLE_KEY`, `GOOGLE_SECRET`, `EMAIL_USER`, `EMAIL_PASSWORD`.

**DEBUG**: `os.getenv('DEBUG', 'True') == 'True'` — defaults to `True` if unset.

**Secrets in code**: none hardcoded for production use.
- `SECRET_KEY` has a fallback literal `'django-insecure-local-dev-only-key-do-not-use-in-production'`, but only when `DEBUG` is also true; if `DEBUG=False` and `SECRET_KEY` is unset, `ImproperlyConfigured` is raised instead of silently using an insecure key.
- `backend/.env` (present in the working tree, not committed — repo-ignored) currently contains a placeholder `SECRET_KEY=gfdsgheytwtwyeyrurururetwtrw` and blank OAuth/email credentials.
- When `DEBUG=False`, additional hardening is applied: `SECURE_SSL_REDIRECT`, `SESSION_COOKIE_SECURE=True`, `CSRF_COOKIE_SECURE=True`, `SECURE_HSTS_SECONDS`, `SECURE_HSTS_INCLUDE_SUBDOMAINS=True`, `SECURE_CONTENT_TYPE_NOSNIFF=True`.

**Auth backends**: GitHub OAuth2, Google OAuth2 (both via `social_core`), plus Django's default `ModelBackend`.

---

## 9. Dataset files in the repo

Single dataset file: **`data/raw/UPI_FRAUD.csv`**.

- **Shape**: 3,075 rows × 12 columns.
- **Columns** (with dtype as read by pandas):
  | Column | dtype |
  |---|---|
  | `Merchant_id` | int64 |
  | `TransactionDate` | float64 (100% null in source data, per code comment in `ml/src/features/schema.py`) |
  | `AverageAmountTransactionDay` | float64 |
  | `TransactionAmount` | float64 |
  | `Is_declined` | object (Y/N) |
  | `TotalNumberOfDeclinesDay` | int64 |
  | `isForeignTransaction` | object (Y/N) |
  | `isHighRiskCountry` | object (Y/N) |
  | `DailyChargebackAvgAmt` | int64 |
  | `Six_MonthAvgChbkAmt` | float64 |
  | `Six_MonthChbkFreq` | int64 |
  | `isFradulent` | object (Y/N) — target column |
- **Class balance**: 2,627 rows `N` (legitimate), 448 rows `Y` (fraudulent).

No other dataset/CSV files exist elsewhere in the repository (`ml/` contains only notebooks, source code, and the trained artifacts — no separate copy of the data).
