# Engineering Audit — Smart UPI Fraud Detection

**Scope:** Full review of the repository as it stands after the Phase 1 structural refactor. This is a read-only assessment — no code changes were made as part of this audit. Every finding below is anchored to a specific file/behavior confirmed during Phase 1 exploration and validation (not a generic checklist).

---

## 1. Repository Architecture

**Current State:** `backend/` (Django), `ml/` (notebooks + artifacts), `data/raw/` (dataset), `docs/` (architecture + thesis), root-level `requirements.txt`/`.env.example`/`.gitignore` — established in this refactor.

**Strengths:** Clear domain separation (web app vs. ML assets vs. docs); single source of truth for the dataset and model artifacts (previously duplicated 3×); no dead directories left over (`DATASET/`, `Finam_document/`, stale `.ipynb_checkpoints/` removed).

**Weaknesses:** `ml/` has no `src/` — training exists only as notebooks, so there is no callable, testable training pipeline. The `users` Django app still carries two unrelated domains (accounts and fraud-prediction) in one app, only separated at the `views/`/`services/` level, not at the Django-app level.

**Risks:** Because the model that's actually served (`upi_fraud_model.pkl`) has no corresponding training script in the repo (only `UPI1.pkl` is traceable to `M6-RFC.ipynb`), the production model cannot currently be regenerated or audited for how it was built.

**Production Readiness Score:** 7/10
**Priority:** Medium
**Recommended Improvements:** Add `ml/src/{preprocessing,train,evaluate}.py` as the canonical, notebook-independent training path; identify/recreate the training code for `upi_fraud_model.pkl`; consider splitting `users` into `accounts` + `fraud_detection` Django apps in a dedicated, carefully-tested migration (out of scope here due to DB/ContentType risk — see Section 20).

---

## 2. Software Architecture

**Current State:** Monolithic Django MVT app. Views split into `views/{auth,prediction,reports,pages}.py`; a thin `services/prediction_service.py` wraps model loading/inference (introduced in Phase 1).

**Strengths:** Clear separation of concerns after the refactor; service layer isolates the one piece of "business logic" (inference) from HTTP concerns.

**Weaknesses:** No repository/data-access layer — views call the Django ORM directly (`Profile.objects.all()`, etc.); `UserPredictModel` + `UserPredictDataForm` exist but are never used by `analyze_upi` (which appends to an in-memory list instead) — a real architecture-drift smell, not just style.

**Risks:** The unused `UserPredictModel` misleads future maintainers into thinking predictions are persisted to the database when they are not (see Section 9/17).

**Production Readiness Score:** 5/10
**Priority:** High
**Recommended Improvements:** Either wire `analyze_upi` to actually save `UserPredictModel` rows (persistent history) or remove the unused model/form to stop the architecture from lying about its own behavior.

---

## 3. Code Quality

**Current State:** Views now organized into small, focused modules (~20-60 lines each) instead of one 206-line file with duplicate/dead imports (both fixed in Phase 1: removed the duplicate `from .models import Profile` and the unused `UserPredictDataForm` import).

**Strengths:** No dead imports left; consistent module boundaries; original logic preserved verbatim.

**Weaknesses:** Naming is inconsistent (`Basic_report`/`Metrics_report` in PascalCase vs. `awareness_page`/`profile_list` in snake_case — a pre-existing PEP8 violation); no type hints anywhere; no docstrings beyond the one added to the new service module; no linter/formatter configuration (no `ruff`/`flake8`/`black`/`pyproject.toml`) to enforce a standard going forward.

**Risks:** Low immediate risk, but code style will drift again without tooling.

**Production Readiness Score:** 5/10
**Priority:** Medium
**Recommended Improvements:** Add `ruff`/`black` with a pre-commit hook; rename `Basic_report`/`Metrics_report` to snake_case (breaking change to the URL name only if renamed carelessly — the view function name can change independently of the URL `name=`).

---

## 4. Machine Learning Pipeline

**Current State:** Six notebooks (`M1`–`M6` + a classification report) perform preprocessing → EDA → training across three algorithms (Naive Bayes, Decision Tree, Random Forest). The artifact that's actually served (`upi_fraud_model.pkl`) is a bundled scikit-learn `Pipeline` (confirmed via unpickle warnings: `StandardScaler`, `OneHotEncoder`, `ColumnTransformer`, then a classifier) — i.e., preprocessing and model are correctly bundled together, which is good practice.

**Strengths:** Preprocessing + model are one artifact (no train/serve skew from separately-applied encoders); the feature set is small and well-defined (9 fields).

**Weaknesses:** The pipeline has no callable, non-notebook entry point; there's no script that reproduces `upi_fraud_model.pkl` end-to-end; each notebook repeats the same `del df['Merchant_id']; del df['TransactionDate']` cleanup instead of a shared function.

**Risks:** Without a reproducible pipeline, the model cannot be retrained if the dataset changes, and no one can verify the exact preprocessing/hyperparameters that produced the artifact currently in production.

**Production Readiness Score:** 3/10
**Priority:** Critical
**Recommended Improvements:** Extract a single `build_pipeline()` + `train()` function used by all notebooks/scripts; commit the exact training script for `upi_fraud_model.pkl`.

---

## 5. Data Pipeline

**Current State:** Single static CSV, `data/raw/UPI_FRAUD.csv` (3,075 rows, 12 columns), read directly by every notebook.

**Strengths:** Small, clean dataset; consistent column names between the CSV, `UserPredictModel`, and the POST payload in `analyze_upi`.

**Weaknesses:** No schema validation (nothing checks that a new CSV drop still has the same 12 columns/dtypes); no data versioning (DVC, checksums); `Merchant_id`/`TransactionDate` are dropped ad hoc per-notebook rather than at ingestion time.

**Risks:** A malformed or updated CSV would silently break every notebook and, indirectly, retraining.

**Production Readiness Score:** 3/10
**Priority:** High
**Recommended Improvements:** Add a lightweight schema check (pandera/pydantic) at load time; consider DVC or a checksum manifest for `data/raw/`.

---

## 6. Feature Engineering

**Current State:** 9 raw features are passed straight through from the HTML form to the model (`AverageAmountTransactionDay`, `TransactionAmount`, `Is_declined`, `TotalNumberOfDeclinesDay`, `isForeignTransaction`, `isHighRiskCountry`, `DailyChargebackAvgAmt`, `Six_MonthAvgChbkAmt`, `Six_MonthChbkFreq`); categorical Y/N fields are encoded inside the pickled pipeline.

**Strengths:** Encoding logic travels with the model artifact, so the Django view doesn't need to duplicate it.

**Weaknesses:** No documented feature dictionary (valid ranges/meaning of each field aren't written down anywhere); no feature-schema check between the Django form and what the model expects — if training code changes column order/names, the app fails silently or loudly with no early warning.

**Risks:** Silent train/serve skew if the model is retrained with a different feature set and the Django form isn't updated in lockstep.

**Production Readiness Score:** 4/10
**Priority:** Medium
**Recommended Improvements:** Add a shared `FEATURE_SCHEMA` (e.g., a list of expected columns + types) imported by both the training code and `prediction_service.py`, with a startup check that the loaded pipeline's expected input matches it.

---

## 7. Model Training

**Current State:** Training happens ad hoc inside notebooks with no fixed `random_state` visible in the reviewed cells, no persisted train/test split, and no centralized metrics log across the three algorithms tried.

**Strengths:** Multiple algorithms were genuinely compared (NB, DTC, RFC), showing real model-selection effort.

**Weaknesses:** Results aren't reproducible (no seed, no saved split); no comparison table/artifact recording which model "won" and why; the notebook that trains the actually-served model isn't identifiable in the repo.

**Risks:** Cannot answer "why was this model chosen" or "can we reproduce this exact model" — both baseline expectations for any production ML system.

**Production Readiness Score:** 2/10
**Priority:** Critical
**Recommended Improvements:** Fix `random_state`; persist the train/test split (or the split indices); save a metrics table (accuracy/precision/recall/F1 per model) as a versioned artifact, not just notebook output.

---

## 8. Model Evaluation

**Current State:** `Classification Report.ipynb` imports `classification_report`, `confusion_matrix`, `accuracy_score`, and `ydata_profiling.ProfileReport`, but the output is only ever rendered inline in the notebook (and, separately, dumped as a static 32k-line HTML file embedded in the Django templates — see Section 21).

**Strengths:** The right evaluation primitives are being used.

**Weaknesses:** No versioned metrics artifact (e.g., a `metrics.json` checked in alongside the model); no held-out test set persisted for future comparison; no cross-validation; no discussion of class imbalance handling (fraud datasets are almost always imbalanced, and nothing in the reviewed cells addresses this explicitly).

**Risks:** Without saved metrics, there's no way to detect model regression when retraining.

**Production Readiness Score:** 3/10
**Priority:** High
**Recommended Improvements:** Persist metrics to a versioned file next to the model artifact; add class-imbalance analysis (class weights / SMOTE / stratified split) and document the decision either way.

---

## 9. Inference Pipeline

**Current State:** `prediction_service.py` loads the model once at import time and exposes `predict(df)`. `analyze_upi` builds a single-row DataFrame from 9 POST fields and calls it synchronously.

**Strengths:** Model loaded once per process (not per-request) — reasonable for this scale; confirmed working end-to-end during Phase 1 validation (`predict()` returned `'N'` for a sample transaction).

**Weaknesses:** Zero error handling — `float(request.POST['TransactionAmount'])` and friends will raise an uncaught `KeyError`/`ValueError` on missing/malformed input, surfacing Django's raw debug error page (since `DEBUG` defaults to `True`); no input validation (a `UserPredictDataForm` already exists in `forms.py` but is unused here — Section 2).

**Risks:** A malformed request from a user (or a bot) crashes the request with a 500 and, in debug mode, could leak internals in the traceback.

**Production Readiness Score:** 3/10
**Priority:** Critical
**Recommended Improvements:** Validate input via the existing `UserPredictDataForm` instead of raw `request.POST[...]` indexing; wrap `predict()` in a try/except that renders a friendly error instead of a 500.

---

## 10. Explainability

**Current State:** None. `result.html` shows only the raw predicted label.

**Strengths:** N/A.

**Weaknesses:** No SHAP/LIME/feature-importance output; no confidence score shown (the model likely supports `predict_proba` given it's a scikit-learn classifier, but it's never called).

**Risks:** For a fraud-detection tool, an unexplained "fraud"/"not fraud" verdict has low practical trust value — this is the single biggest gap from an "AI engineering maturity" standpoint.

**Production Readiness Score:** 1/10
**Priority:** Medium (High if this is meant to demonstrate applied ML maturity for a portfolio)
**Recommended Improvements:** Surface `predict_proba` as a confidence score; add a simple feature-contribution breakdown per prediction (even a basic coefficient/importance display, not full SHAP, would meaningfully improve this).

---

## 11. Model Persistence

**Current State:** `ml/artifacts/models/upi_fraud_model.pkl` (1.7 MB) and `UPI1.pkl` (820 KB, unused by the app) committed directly to the repo.

**Strengths:** Single canonical location post-refactor (previously triplicated across `DATASET/`, `Deployment/users/`).

**Weaknesses:** No model card / metadata (training date, metrics, library versions); **confirmed sklearn version mismatch during validation** — the pickled `StandardScaler`/`OneHotEncoder`/`ColumnTransformer`/classifier/`Pipeline` were trained under scikit-learn 1.2.1 but the repo's own `requirements.txt` (even after this refactor's fix) pins 1.1.3, producing explicit `UserWarning`s on every load about potential breaking behavior.

**Risks:** This is a live reproducibility bug, not a hypothetical one — it was directly observed running `manage.py check` in Section "Validation." Predictions could subtly differ from what the model would produce under its original training environment.

**Production Readiness Score:** 3/10
**Priority:** Critical
**Recommended Improvements:** Pin `scikit-learn` to the exact version the model was trained with (retrain and re-pin, or determine the original version and match it); add a `model_metadata.json` (training date, library versions, metrics) alongside the `.pkl` files; for a real production system, move artifacts out of git into an artifact store (S3/MLflow/DVC) — acceptable to keep in git for a portfolio repo, but worth noting as a scaling limit.

---

## 12. Error Handling

**Current State:** No `try`/`except` blocks exist anywhere in `views/` or `services/`. No custom `404.html`/`500.html` templates.

**Strengths:** N/A — this is a clean gap, not a partially-done one.

**Weaknesses:** Every external input path (form submissions, file uploads, the prediction form) assumes well-formed data.

**Risks:** Any malformed request currently produces Django's raw debug traceback page (since `DEBUG` defaults `True`), which is both a poor user experience and, if ever deployed with `DEBUG=True`, a real information-disclosure risk (Section 15).

**Production Readiness Score:** 2/10
**Priority:** Critical
**Recommended Improvements:** Add form validation (Section 9), custom error templates, and a top-level exception-handling strategy (Django's `handler500`, or middleware that logs and returns a generic error page).

---

## 13. Logging

**Current State:** No `LOGGING` configuration in `settings.py`; no use of Python's `logging` module anywhere in the codebase.

**Strengths:** N/A.

**Weaknesses:** The only visibility into what the app is doing is the raw `runserver` console output and Django's default request line logging.

**Risks:** No way to audit predictions made, errors encountered, or login attempts in anything beyond a local dev console — unusable for any real deployment.

**Production Readiness Score:** 1/10
**Priority:** High
**Recommended Improvements:** Add a `LOGGING` dict in `settings.py` (even a simple file/console handler); log each prediction request (input + result) for auditability, given this is a fraud-detection tool.

---

## 14. Configuration Management

**Current State:** Improved substantially in this refactor: `SECRET_KEY`/OAuth keys/email credentials via `.env` (unchanged from before, now documented in `.env.example`); `DEBUG` and `ALLOWED_HOSTS` are now environment-driven (previously hardcoded); `MODEL_ARTIFACT_DIR` is now configurable and resolves robustly via `BASE_DIR` instead of a CWD-relative string.

**Strengths:** No more hardcoded, CWD-dependent paths; single `.env.example` documents every required variable.

**Weaknesses:** `SECRET_KEY = str(os.getenv('SECRET_KEY'))` still silently becomes the literal string `"None"` if the env var is missing — the app boots "successfully" with a known, insecure secret key rather than failing fast; no environment-specific settings split (dev/staging/prod all share one `settings.py`).

**Risks:** A misconfigured deployment (missing `.env`) fails silently into an insecure state instead of refusing to start.

**Production Readiness Score:** 6/10
**Priority:** High
**Recommended Improvements:** Raise `ImproperlyConfigured` if `SECRET_KEY` is unset instead of defaulting to `"None"`; consider `settings/base.py` + `settings/{dev,prod}.py` for environment-specific overrides.

---

## 15. Security

**Current State:** Django's built-in auth, CSRF middleware, and password validators are in place and functioning (CSRF was confirmed active during Phase 1 validation — an unauthenticated raw POST to `/analyze_upi/` correctly returned `403`).

**Strengths:** CSRF protection works; password validators configured; OAuth via `social-auth-core` avoids storing third-party credentials directly.

**Weaknesses / Risks (concrete, not hypothetical):**
- **`SECRET_KEY` insecure fallback** (Section 14).
- **`DEBUG` defaults to `True`** — would leak stack traces/environment details if deployed without explicitly overriding.
- **No `SECURE_*` settings** — `SECURE_SSL_REDIRECT`, `SESSION_COOKIE_SECURE`, `CSRF_COOKIE_SECURE`, `SECURE_HSTS_SECONDS` are all unset.
- **`/database/` exposes every user's submitted transaction data to any logged-in user** — `model_db_view` renders the entire in-memory `user_db` list with no per-user filtering or object-level permission check. This is a real data-exposure bug: User A can see User B's submitted (albeit synthetic/demo) transaction data and predictions.
- No rate limiting on login/register/prediction endpoints.
- Pickle-based model loading is a known deserialization-risk pattern in general (not exploitable here since the `.pkl` is first-party, but worth noting for a portfolio write-up).

**Production Readiness Score:** 3/10
**Priority:** Critical
**Recommended Improvements:** Fix the `/database/` data-exposure bug first (filter by request.user or remove the shared list entirely); set `DEBUG=False` and the `SECURE_*` settings for any non-local deployment; fail fast on missing `SECRET_KEY`.

---

## 16. Performance

**Current State:** Model loaded once per process (good). `Basic_report.html` is a **32,138-line** vendored HTML file (bootstrap/jQuery source inlined, produced by a profiling tool) served through Django's template engine on every request to `/Basic_report/`.

**Strengths:** No per-request model reload; static assets otherwise reasonably small.

**Weaknesses:** Serving a 32k-line static report through the Django template engine (parse + render on every request) instead of as a plain static file is pure wasted CPU/memory per request; no caching layer configured at all (no `CACHES` beyond Django defaults); no gzip/compression middleware.

**Risks:** `/Basic_report/` is a real, measurable performance liability under any load — likely the single most wasteful endpoint in the app relative to what it delivers.

**Production Readiness Score:** 3/10
**Priority:** High
**Recommended Improvements:** Serve `Basic_report.html` as a static file (via `whitenoise` or the static file server) instead of through `render()`; add `django.middleware.gzip.GZipMiddleware`.

---

## 17. Scalability

**Current State:** SQLite database; in-memory `user_db = []` list for prediction history; single-process dev server.

**Strengths:** Entirely adequate for a single-instance demo/portfolio deployment.

**Weaknesses:** The in-memory `user_db` list is **process-local** — under any multi-worker deployment (gunicorn with >1 worker, multiple pods, etc.) each worker has its own independent list, so `/database/` would show inconsistent, partial history depending on which worker handled which request. SQLite does not support concurrent writers well and isn't viable beyond single-instance use.

**Risks:** The app would not behave correctly the moment it's scaled horizontally — this isn't a future concern, it's already broken in spirit for anything beyond `runserver`.

**Production Readiness Score:** 2/10
**Priority:** High (Critical if any multi-instance deployment is planned)
**Recommended Improvements:** Persist predictions to `UserPredictModel` (a real DB table) instead of an in-memory list — this single change fixes both the scalability gap and the architecture-drift issue in Section 2; migrate off SQLite to Postgres for any real deployment.

---

## 18. Testing

**Current State:** **Zero automated tests exist in the repository** — there isn't even a `tests.py` stub in the `users` app, and no test runner configuration (no `pytest.ini`, `tox.ini`, or CI workflow) was found anywhere in the repo.

**Strengths:** N/A.

**Weaknesses:** No unit tests for forms/views/models; no test for `prediction_service.predict()` against known input/output pairs; no test confirming the model artifact loads successfully (which would have caught the sklearn-version mismatch in Section 11 automatically, in CI, before deployment).

**Risks:** Any future change — including this very refactor — has no automated safety net; correctness was verified manually in this session instead of via a repeatable test suite.

**Production Readiness Score:** 0/10
**Priority:** Critical
**Recommended Improvements:** Add `users/tests/` with at minimum: a model-loading smoke test, a `predict()` regression test with fixed inputs, and view-level tests for the auth flows using Django's test client; wire into a CI workflow (GitHub Actions) that runs on every push.

---

## 19. Documentation

**Current State:** Prior to this refactor, `README.md` was the **generic boilerplate from the original forked "Django-registration-and-login-system" template** — it never mentioned UPI or fraud detection at all. This refactor replaced it with a project-accurate `README.md` plus `docs/architecture.md` and this `docs/AUDIT.md`.

**Strengths:** Setup instructions, repo layout, and architecture are now accurately documented and match the real codebase.

**Weaknesses:** No docstrings in the source code; no data dictionary describing the 9 model features and their valid ranges; no `LICENSE` file (none exists); no `CONTRIBUTING.md`.

**Risks:** Low, mostly a portfolio/professionalism gap rather than a functional one.

**Production Readiness Score:** 6/10 (post-refactor; was ~1/10 before)
**Priority:** Medium
**Recommended Improvements:** Add a `LICENSE`; add a short data dictionary for the 9 features; add docstrings to `services/prediction_service.py` and the view modules.

---

## 20. Django Integration

**Current State:** Django 4.1.2 (released October 2022) — now multiple minor versions behind current Django LTS, meaning it no longer receives security patches. `social-auth-app-django`/`social-auth-core` integrate GitHub/Google OAuth correctly. `UserPredictModel` + `UserPredictDataForm` are defined and even registered in `INSTALLED_APPS`'s migration history, but never used by the actual prediction flow (Section 2).

**Strengths:** Django fundamentals (CSRF, auth, migrations, admin) are used correctly and idiomatically; the app split (single `users` app with an internal `views/`/`services/` package structure) was deliberately kept in this refactor rather than restructured into multiple Django apps, specifically to avoid unsafe changes to `django_content_type`/migration history against the existing local `db.sqlite3` (see `docs/architecture.md`).

**Weaknesses:** Outdated Django version; `admin.py` only registers `Profile`, not `UserPredictModel` (further evidence it's vestigial).

**Risks:** Running an EOL-adjacent Django version in anything internet-facing is a real security exposure.

**Production Readiness Score:** 5/10
**Priority:** High
**Recommended Improvements:** Upgrade to the current Django LTS in a dedicated, tested change (out of scope for this structural refactor); decide the fate of `UserPredictModel` (wire it up or remove it).

---

## 21. Frontend Integration

**Current State:** Two incompatible styling strategies coexist: `users/base.html` (used by all `app/*` fraud-detection pages) pulls Bootstrap 5/Font Awesome from CDN with a 679-line inline `<style>` block; `users/home.html` instead hardcodes local `/static/css/bootstrap.min.css` paths (with an unused `{% load static %}` tag) and carries its own separate 1,333-line inline stylesheet. `app/Basic_report.html` is a 32,138-line vendored HTML dump (see Section 16).

**Strengths:** Visually polished, custom-themed UI (not just default Bootstrap) — genuinely above-average for a college project.

**Weaknesses:** No shared design system between the two template families (duplicate CSS custom-property definitions); no CSS build pipeline/bundler; no JS framework — all interactivity is inline `<script>` blocks per template.

**Risks:** Low functional risk, high maintenance cost — any design-token change (a color, a spacing value) has to be manually propagated across every template's inline `<style>` block.

**Production Readiness Score:** 4/10
**Priority:** Medium
**Recommended Improvements:** Extract the repeated CSS custom properties into one shared stylesheet included by both template families; standardize on either CDN or local static assets, not both.

---

## 22. Deployment Readiness

**Current State:** No `Dockerfile`/`docker-compose.yml`; no production WSGI server in `requirements.txt` (`gunicorn`/`uwsgi` absent — only Django's dev server is runnable as-is); no CI/CD (`.github/workflows/` doesn't exist for this repo); no health-check endpoint.

**Strengths:** `requirements.txt` now actually installs cleanly and the app boots and serves correctly (validated in this session on Python 3.11 with a clean venv).

**Weaknesses:** `DEBUG` defaults `True`; static/media handling relies entirely on Django's dev-server file serving (fine for `DEBUG=True`, not for production without `whitenoise`/a CDN/nginx in front).

**Risks:** The app is runnable locally but not deployable as-is to a real host without meaningful additional work.

**Production Readiness Score:** 2/10
**Priority:** High
**Recommended Improvements:** Add a `Dockerfile` + `gunicorn`; add `whitenoise` for static file serving; add a minimal CI workflow (install deps, run `manage.py check`, run tests once they exist).

---

## 23. AI/ML Engineering Best Practices

**Current State:** No experiment tracking (MLflow/Weights & Biases); no model registry/versioning beyond a filename; no monitoring or drift detection; no automated retraining; the entire training pipeline exists only as manually-run Jupyter notebooks, with the notebook that produced the actually-served model artifact untraceable in the repo (Section 4/7/11).

**Strengths:** The one thing done well — bundling preprocessing + model into a single scikit-learn `Pipeline` artifact — avoids the most common train/serve skew failure mode.

**Weaknesses:** Everything else expected of a production ML system is absent: no reproducible training entry point, no persisted metrics, no data/model versioning, no CI for the ML side.

**Risks:** This is the largest single gap in the repository from a "senior AI engineer portfolio" perspective — the web engineering is now solid post-refactor, but the ML engineering maturity lags well behind it.

**Production Readiness Score:** 2/10
**Priority:** Critical
**Recommended Improvements:** This is the highest-leverage area for future work — see Phase A/B/C roadmap below. At minimum: a callable training script, persisted metrics, and a fixed random seed would close most of the gap cheaply.

---

# Final Report

## 1. Old vs New Repository Structure

**Before:**
```
Smart-upi-fraud-detect/
├── .ipynb_checkpoints/            # Jupyter clutter, orphaned
├── DATASET/                        # notebooks + CSV + 2 duplicate .pkl files (+ its own .ipynb_checkpoints)
├── Deployment/
│   ├── user_management/            # Django project settings
│   ├── users/                       # Django app: auth AND fraud-prediction views in one 206-line views.py
│   │   ├── upi_fraud_model.pkl       # duplicate of DATASET's copy
│   │   ├── UPI1.pkl                   # duplicate, unused
│   │   └── static/                    # duplicate of Deployment/static/
│   ├── static/                       # duplicate of users/static/ (stale collectstatic output)
│   ├── requirements.txt              # missing pandas/numpy/scikit-learn/joblib/matplotlib/seaborn
│   └── README.md                     # generic unrelated Django tutorial boilerplate
└── Finam_document/                  # thesis Word/PPT at repo root
```

**After:**
```
Smart-upi-fraud-detect/
├── .gitignore / .env.example / README.md / requirements.txt   # project-level, accurate
├── docs/
│   ├── architecture.md
│   ├── AUDIT.md
│   └── thesis/                       # UPI-FINAL.docx / .pptx
├── data/raw/UPI_FRAUD.csv             # single canonical copy
├── ml/
│   ├── notebooks/                     # M1-M6 + Classification Report, paths fixed to new data location
│   └── artifacts/models/              # single canonical upi_fraud_model.pkl + UPI1.pkl
└── backend/
    ├── manage.py
    ├── config/                         # was user_management/
    └── users/
        ├── views/{auth,prediction,reports,pages}.py   # was one 206-line views.py
        ├── services/prediction_service.py               # new: isolated inference logic
        ├── static/                                        # deduplicated (root-level copy removed)
        └── templates/, models.py, forms.py, migrations/... (unchanged)
```

## 2. Architecture Decisions Made and Why

- **`Deployment/` → `backend/`, `user_management/` → `config/`:** standard, unambiguous naming; zero functional risk (pure import-path renames, verified via `manage.py check`).
- **Kept `users` as a single Django app; split internally into `views/`+`services/` instead of separate Django apps:** explicitly chosen over a cleaner-looking `accounts`/`fraud_detection` app split because moving models across Django apps rewrites migration history and `django_content_type` rows against the existing local `db.sqlite3`, with no project-scoped git history to fall back on if it broke local logins. Approved by the user during planning.
- **`MODEL_ARTIFACT_DIR` setting instead of a hardcoded path:** the original `joblib.load('users/upi_fraud_model.pkl')` only worked by accident (relied on the server's CWD). Moving the artifact required fixing this regardless; making it an env-configurable setting was the natural, low-risk way to do it.
- **Single `ml/` and `data/` directories instead of nesting under `backend/`:** the dataset and notebooks are conceptually independent of the Django app (a future retraining pipeline shouldn't need to know about Django at all).
- **No git repository initialized:** the project currently sits inside an unrelated, accidental repo rooted at the whole home directory; the user explicitly chose to leave git alone for this task.

## 3. Files Moved

| From | To |
|---|---|
| `Deployment/` (entire tree) | `backend/` |
| `Deployment/user_management/` | `backend/config/` |
| `Deployment/users/views.py` | `backend/users/views/{auth,prediction,reports,pages}.py` |
| `DATASET/UPI_FRAUD.csv` | `data/raw/UPI_FRAUD.csv` |
| `DATASET/*.ipynb` (6 files) | `ml/notebooks/` (read_csv paths updated) |
| `DATASET/upi_fraud_model.pkl` | `ml/artifacts/models/upi_fraud_model.pkl` |
| `DATASET/UPI1.pkl` | `ml/artifacts/models/UPI1.pkl` |
| `Finam_document/UPI-FINAL.{docx,pptx}` | `docs/thesis/` |

## 4. Files Removed

- `.ipynb_checkpoints/` (root) and `DATASET/.ipynb_checkpoints/` — Jupyter auto-save clutter.
- All `__pycache__/` directories.
- `Deployment/users/upi_fraud_model.pkl`, `Deployment/users/UPI1.pkl` — duplicates of the `DATASET/` copies.
- `Deployment/static/` — byte-identical duplicate of `Deployment/users/static/` (confirmed via `diff -rq` before deletion), and a stale committed `collectstatic` output.
- `Deployment/requirements.txt`, `Deployment/README.md`, `Deployment/.gitignore` — superseded by consolidated root-level versions.
- Empty `DATASET/` and `Finam_document/` directories once emptied.

## 5. Structural Improvements Achieved

- `requirements.txt` now actually installs and runs the app (previously missing 6 required packages, confirmed by a clean `pip install` + `manage.py check` + live `runserver` + an end-to-end prediction in this session).
- Zero duplicate binary/static assets remain (was: 3 copies of the model, 2 copies of the static folder).
- Fragile CWD-relative model path replaced with a robust, configurable one.
- Dead code removed (duplicate import, unused import) during the views split, with no behavior change (verified via `makemigrations --check` showing no drift and a live prediction test returning the correct result).
- Documentation now accurately describes this project (previously an unrelated generic template README).

## 6. Remaining Structural Issues

- Training code for the actually-served model (`upi_fraud_model.pkl`) is not traceable to any notebook in the repo.
- `users` app still mixes two domains (deliberately, for DB safety — see Section 20/2 above).
- No tests, no CI, no logging, no Docker (Sections 12-13, 18, 22).
- `sklearn` version mismatch between training (1.2.1) and the pinned runtime (1.1.3) is a live, observed warning, not fixed in this refactor (fixing it would mean either retraining or guessing at a version bump, both outside "don't touch the ML pipeline").

## 7-12. Scores

| Score | Value | Rationale |
|---|---|---|
| **Overall Repository Score** | **7/10** | Structure, hygiene, and documentation are now genuinely production-grade; the underlying app's technical debt (no tests, no logging) caps it below 8. |
| **Software Engineering Score** | **5/10** | Clean modularity and config management now; held back by zero tests, zero error handling, zero logging. |
| **Machine Learning Engineering Score** | **3.5/10** | One thing done right (bundled pipeline artifact); everything else — reproducibility, versioning, metrics, experiment tracking — is missing. |
| **Production Readiness Score** | **3/10** | Confirmed-working app, but insecure defaults (`DEBUG=True`, weak `SECRET_KEY` fallback), a real data-exposure bug (`/database/`), and a scalability-breaking in-memory store. |
| **Resume Value Score** | **7/10** | Post-refactor, this reads as a legitimate full-stack + applied-ML portfolio piece with a working demo, clean structure, and honest documentation — noticeably above a typical college fraud-detection repo. |
| **AI Engineer Portfolio Score** | **6/10** | Good demonstration of shipping an ML model behind a real web app; adding a reproducible training script, persisted metrics, and a couple of tests would be the highest-leverage next step to stand out further. |

## 13. Prioritized Implementation Roadmap

**Phase A — Critical**
1. Fix the `/database/` data-exposure bug (any logged-in user sees every user's prediction history).
2. Add input validation + error handling to `analyze_upi` (use the existing, unused `UserPredictDataForm`).
3. Add a minimal test suite (model-load smoke test + `predict()` regression test at minimum) and wire it into CI.
4. Recover or recreate the training script for `upi_fraud_model.pkl`; fix the scikit-learn version mismatch (retrain or re-pin to the matching version).
5. Fail fast on a missing `SECRET_KEY` instead of silently booting with `"None"`.

**Phase B — High Impact**
6. Persist predictions to `UserPredictModel` instead of the in-memory `user_db` list (fixes both the architecture drift in Section 2 and the scalability bug in Section 17).
7. Add logging (`LOGGING` config + request/prediction audit logging).
8. Serve `Basic_report.html` as a static file instead of through the Django template engine.
9. Add `SECURE_*` settings and set `DEBUG=False` for any non-local deployment; upgrade Django off the current EOL-adjacent version.
10. Add a `Dockerfile` + `gunicorn` + a basic CI workflow.

**Phase C — Nice to Have**
11. Add model explainability (confidence score via `predict_proba`, basic feature-importance display).
12. Add experiment tracking / persisted metrics for the ML side.
13. Add a linter/formatter (`ruff`/`black`) and docstrings.
14. Unify the two frontend styling strategies into one shared stylesheet.
15. Add a `LICENSE` and a data dictionary for the 9 model features.
