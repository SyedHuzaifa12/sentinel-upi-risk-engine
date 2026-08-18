# Render deployment notes

**LIVE: https://sentinel-upi-risk-engine.onrender.com** (deployed
successfully 2026-08-19, after the `libgomp1` fix below).

Two deploy targets now exist. **docker-compose.yml / Dockerfile** (postgres,
redis, api, worker, ui, generator) is untouched and remains the primary
local-dev target. **Dockerfile.render / render.yaml / render_app.py /
render_seed.py** are an ADDED second target: a single-process, single-
container deploy to Render's free tier (512MB RAM, 0.1 CPU, sleeps after
15 min idle, 750 instance-hours/month, 500 build minutes, no card). Nothing
in the first target was deleted or modified to build the second.

## Post-deploy fixes (2026-08-19, found from real usage of the live site)

1. **"Replay N events" 500 — root cause was a deterministic generator bug,
   not memory/timeout.** `ml/src/generator/population.py::build_payees`
   crashes with `ValueError: low >= high` whenever called with `days <= 1`
   (the churn-window math produces `window_end < window_start`) — the
   replay view called `generate(days=1, ...)`, so this fired on literally
   every click. Found by running the actual replay code path locally
   rather than theorizing from the traceback alone. Fixed in the generator
   itself (`days <= 1` is now a defined degenerate case, not a crash) plus
   the caller now uses `days=3`. Separately hardened regardless: the
   synchronous view became a background-thread + polling design (25
   events/click, not 100), wrapped in try/except/finally so any future
   failure surfaces in a JSON `error` field instead of a bare 500, and
   SHAP's TreeExplainer construction is now pre-warmed once at startup
   (`reason_codes.warm_up_explainers()`) instead of happening
   unpredictably on the first live non-ALLOW decision. See PROGRESS.md's
   Phase 8 section for full detail; regression test in
   `ml/tests/test_generator.py`.
2. **Bank auto-detect fired on `blur` only** — a user who never left the
   VPA field before submitting saw the stale `OTHER` default even though
   the suffix WAS correctly mapped. Fixed by also triggering on `input`;
   expanded the mapping to the additional handles named in the report.
3. **Reason codes reviewed for actionability**, not just correctness — see
   PROGRESS.md's Phase 8 section for the specific rewrites.
4. **Landing page**: banner tone softened (dropped the self-deprecating
   framing, kept the synthetic-data disclosure itself prominent) and
   visually redesigned to reuse the app's existing card/pill/icon language
   instead of a one-off style.
5. **Full dark-theme redesign + link cleanup** (2026-08-19, supersedes
   item 4's light-blue pass): every page (landing, monitoring, review
   queue, sandbox, result) now shares one stylesheet
   (`backend/users/static/css/design-system.css`) and an identical nav bar;
   the footer/Results-section links no longer name internal filenames or
   point at a broken bare `https://github.com/`. FastAPI's `/api/docs` is
   dark-themed too via a custom `/docs` route in `service/main.py`
   (`docs_url=None` + a CSS override injected into the stock Swagger HTML)
   with a "Back to dashboard" link — kept, not removed, as documented
   technical evidence. No deploy-relevant code changed (no new
   dependencies, no env vars, no Dockerfile/render.yaml changes) — this is
   templates/CSS/one FastAPI route only. See PROGRESS.md's Phase 8 section,
   item 6, for full detail including the icon-name bugs caught before
   shipping.

## Architecture: one process, two apps

`render_app.py` composes Django (WSGI) and the FastAPI scoring service
(ASGI) into one Starlette app, served by one `uvicorn` process:

```python
app = Starlette(
    routes=[
        Mount("/api", app=fastapi_app),        # same app as `uvicorn service.main:app`
        Mount("/", app=WSGIMiddleware(django_wsgi_app)),
    ],
    lifespan=combined_lifespan,
)
```

**The one previously-unverified risk was checked and fixed before writing
any of this**: Starlette's `Mount()` does **not** forward ASGI lifespan
startup/shutdown into a mounted sub-app (confirmed empirically with a
minimal repro — a FastAPI sub-app's own `lifespan=` context manager never
ran when mounted this way; `scoring.load_models()` would never have
executed, and every `/api/v1/score` call would have 500'd with "load_models()
must run before score_event()"). The fix: `render_app.py`'s own
`combined_lifespan` explicitly delegates to `service.main`'s **existing**
`lifespan(app)` function, called against the mounted `fastapi_app` instance
— zero duplicated startup logic, and confirmed working end-to-end with a
`TestClient` smoke test: `GET /api/v1/health`, `GET /api/v1/model-info`,
`GET /api/docs`, `GET /login/`, and `GET /` (the landing page) all return
200 with real data, models loaded, against the combined app.

Also confirmed: `request.app` inside a route handler mounted this way
correctly resolves to the sub-app itself (not the parent), so
`service/routes.py`'s existing `request.app.state.store` /
`.decision_log` / `.model_info` reads work completely unchanged.

**Worker placement**: `worker/consumer.py`'s Redis Streams consumer is
**not** part of this deploy target at all — `STORE_BACKEND=memory` (no
Redis), and nothing in `render_app.py`'s or `render_seed.py`'s import chain
touches `worker/` or `redis` (checked directly: `service/deps.py`'s
`build_store()` only imports `redis_store` inside the `if backend ==
"redis":` branch, never reached when `STORE_BACKEND=memory`). Live-feeling
traffic comes from `render_seed.py` at startup and the monitoring
dashboard's "Replay N events" button at runtime instead — both score
events **in-process**, directly, no stream/queue involved.

## Sandbox scoring: unchanged code, just a loopback URL

`backend/users/services/prediction_service.py` already calls
`settings.RISK_API_URL` + `/v1/score` over HTTP (Phase 4's design — Django
is a client of the scoring service, not a model host). On Render, the
entrypoint script sets `RISK_API_URL=http://127.0.0.1:$PORT/api` at
container start (computed then, not at build time, since `$PORT` is only
known once Render assigns it) — so the sandbox form just calls itself over
loopback at the `/api` mount point. **Zero changes needed** to
`prediction_service.py` or `views/prediction.py`.

## decisionlog: Neon, not Render's free Postgres, not SQLite

`decisionlog/writer.py` hardcodes `import psycopg` (psycopg3, Postgres wire
protocol only) — it cannot open a SQLite file at all, and
`decisionlog/schema.py`'s append-only trigger is Postgres-specific
`plpgsql` besides. **Not fixed, deliberately**: building a second
SQLite-backed `DecisionLog` + schema is real, avoidable work; the whole
point of `decisionlog` is DB-enforced immutability, easiest to keep exactly
where it's already proven correct.

**Do not use Render's own free Postgres tier — it expires 30 days after
creation** and would permanently break the demo link. Use
[Neon](https://neon.tech) (permanent free tier) instead; Django and
`decisionlog` both connect to the same Neon database via one
`DATABASE_URL`. Neon's dashboard-provided connection strings already
include `?sslmode=require`; use it as-is, no code change needed (psycopg3
and psycopg2 both honor the `sslmode` query parameter natively).

## Startup seeding: real, unverified timing risk

`render_seed.py` runs once per container start (idempotent — skips if
`decisions` already has >= 250 rows). **Reduced from an original
2000-generated/600-scored design** (2026-08-17): 0.1 CPU makes scoring the
slow part, and a startup timeout fails the whole deploy, not just the seed
— 300 decisions is already plenty to populate the monitoring dashboard and
review queue. Measured **locally, full CPU, no network**: `load_models()`
~4.6s, generating 1000 events ~0.4s, scoring 300 of them (SHAP included)
~4.5s — about 9.5s end to end. **This may still be slower on Render**:
0.1 vCPU is a tenth of what this was measured on, and each of the 300
`decisionlog.record()` calls is a real network round-trip to Neon that this
local measurement doesn't include at all.

Two safety valves, not just optimism:
1. A hard wall-clock cutoff (`RENDER_SEED_BUDGET_S`, default 45s) stops
   scoring early rather than risk blowing the whole startup budget — a
   partial seed (fewer than 250 rows) just re-attempts seeding on the next
   restart, nothing is lost.
2. `RENDER_SEED_N_SCORE` / `RENDER_SEED_N_GENERATE` env vars let this be
   tuned down further after watching the first deploy's actual logs,
   without a code change.

**Watch the first deploy's logs for `render_seed:` lines** — if it hits the
45s budget consistently, lower `RENDER_SEED_N_SCORE` to 100-150 in the
Render dashboard and redeploy (no rebuild needed, just a restart, since
it's an env var).

## SHAP: lazy-loaded, not eagerly loaded at import time

`ml/src/policy/reason_codes.py`'s `shap.TreeExplainer` construction (real,
measurable startup cost) is now built on first actual use, not at module
import time — reverses a Phase 3 decision, made explicitly for this deploy
target's slow-CPU startup constraint (see that module's own comment for the
full reasoning). Cold model + warm model themselves are still loaded
eagerly (cheap `joblib.load`, not the slow part). This is a shared-code
change, not Render-specific — it benefits the docker-compose target's
`api`/`worker` cold-start time too, with no behavior change once the
explainer is actually needed.

## Monitoring dashboard: "Replay N events" button

`backend/users/views/monitoring.py`'s `replay_events` view scores 100 fresh
synthetic events in-process against the shared store/decision log already
loaded by `render_app.py`'s combined lifespan — reached via
`service.main.app.state`, the same singleton object `render_app.py`
mounted. **Gracefully degrades** on the docker-compose target (where
`service.main.app`'s lifespan never ran in the `ui` container's own
process): the button renders disabled with a clear "not available in this
deployment" message rather than a 500 — checked via `hasattr(state,
"store")`, not assumed. Each click uses a small (15 payers / 8 payees) pool
so payees recur *within* the 100-event batch, giving realistic warm-model
routing without needing to persist a separate identity pool across clicks;
seeded from the current timestamp so repeated clicks show different
transactions (a deliberate exception to this project's usual
"always a fixed seed" rule — a live demo button is a different use case
than training/backtesting reproducibility).

## Single-process RSS (unchanged finding, still holds)

Live-measured (`docker stats`, the docker-compose target's containers):
api 206.8 MiB, ui 144.1 MiB, worker 201.5 MiB. `ui`'s Django process
imports **only** `numpy` directly (for ULID generation) — none of
pandas/lightgbm/scikit-learn/shap. So merging processes does not
deduplicate an ML-library copy that doesn't currently exist twice; the real
saving is one Python interpreter instead of two. **Estimated combined RSS:
~260-320MB** — real headroom under 512MB, but **CPU (0.1 vCPU) is the more
likely actual constraint**, not RAM, given SHAP's measured ~20.9ms p99 cost
per non-ALLOW decision on dedicated local hardware (see the seeding timing
risk above for the concrete version of this concern).

## One pre-existing, unrelated environment finding

`requirements.txt` pins `Django==4.1.2`; this dev machine's `.venv` has
`Django==5.2.17` installed (drifted independently at some point, not
something this session changed). `Dockerfile.render` builds from
`requirements.txt` fresh, so the actual Render deploy gets 4.1.2, not
whatever's in this local venv — the `TestClient` verification above ran
against 5.2.17, one version-major ahead of what will actually deploy.
Django 4.1 already has `get_asgi_application()` and the
`CSRF_TRUSTED_ORIGINS` wildcard syntax used in `settings.py`, so this
shouldn't matter, but it's a real, unverified-on-4.1.2 gap worth knowing
about rather than silently assuming away. **Since resolved**: installed the
actual pinned `Django==4.1.2` locally and re-ran every check above against
it directly — identical results to 5.2.17 (see the incident log in
PROGRESS.md's Phase 8 section for the exact commands). No longer a gap.

## Incident: first real deploy attempt failed — `libgomp.so.1` missing

**What happened**: migrations applied cleanly against Neon (31/31), the
image built successfully, but `render_seed.py` crashed at container start
with `OSError: libgomp.so.1: cannot open shared object file` at `import
lightgbm`.

**Root cause**: LightGBM's compiled extension dynamically links
`libgomp.so.1` (OpenMP) at import time — it does not vendor this inside its
own wheel, unlike numpy/scipy/pandas, which bundle their own
OpenBLAS/runtime libs and need nothing extra. `python:3.11-slim` doesn't
ship `libgomp1`. The plain `Dockerfile` (docker-compose target) never hit
this because it never strips `build-essential` out of its single stage —
installing `gcc` pulls in `libgomp1` as an *incidental* transitive
dependency, silently masking the real runtime requirement the whole time.
`Dockerfile.render`'s multi-stage build correctly stripped that accidental
dependency along with the genuine build tools — which is exactly why this
gap became visible instead of staying hidden forever.

**Fix**: added `libgomp1` to the FINAL stage's `apt-get install` line (not
the builder stage — it needs to exist at runtime, not build time).
**Audited, not assumed**, whether anything else needed a similar add:
scikit-learn's and shap's compiled extensions can also use OpenMP in some
builds, but that's the *same* `libgomp.so.1`, already covered by this one
package. `psycopg2-binary`/`psycopg[binary]` vendor `libpq` inside their own
wheels (that's what "binary" means). `cryptography`'s OpenSSL need is
already satisfied by `python:3.11-slim`'s own base packages — proven by the
plain `Dockerfile` using the identical base image successfully today.

**Also hardened**: `render_seed.py`'s `if __name__ == "__main__":` block now
wraps `main()` in a try/except that prints one unambiguous `render_seed:
FATAL -- ...` line before the traceback, and `main()`'s body uses
try/finally so the Neon connection is released even on a mid-loop failure.
This does **not** and cannot catch an import-time crash like the libgomp
one (the failing `import lightgbm` line runs before this block exists at
all) — that class of failure has exactly one real fix, which is not letting
the import fail in the first place (the apt package above). What this
hardening actually buys: a readable single-line failure for any *runtime*
seeding failure (a dropped Neon connection, a scoring error), verified
directly with a deliberately-broken `DATABASE_URL` locally — clean FATAL
message, then the traceback, real process exit code 1, `set -e` in
`render_entrypoint.sh` already stops the script before ever reaching `exec
uvicorn`. The "ran three times" in the failed deploy's logs was Render's own
container-restart-on-crash-loop policy (platform-level, not something this
repo's scripts control) retrying a *deterministic* failure — it stops
recurring on its own once the underlying crash is actually fixed, which is
what the apt package does.

---

## Click-by-click Render setup

### 1. Neon database
1. Sign up at [neon.tech](https://neon.tech), create a project (any region
   close to Render's — Render's free tier runs in Oregon, US-West, by
   default).
2. From the Neon dashboard, copy the connection string (it looks like
   `postgresql://<user>:<password>@<host>/<db>?sslmode=require`). Keep it —
   this is `DATABASE_URL`.

### 2. Push this repo to GitHub
Render's Blueprint deploy reads `render.yaml` from a connected GitHub repo.
Commit and push everything in this phase's diff (`Dockerfile.render`,
`render.yaml`, `render_app.py`, `render_seed.py`, `render_entrypoint.sh`,
plus the shared-code changes to `settings.py`/`reason_codes.py`/
`scoring.py`/`monitoring.py`/`urls.py`/`home.html`) before starting the
Render setup below.

### 3. Create the Blueprint on Render
1. [dashboard.render.com](https://dashboard.render.com) → **New** → **Blueprint**.
2. Connect the GitHub repo. Render detects `render.yaml` automatically.
3. It will prompt for the two `sync: false` env vars declared in
   `render.yaml`:
   - `DATABASE_URL` — paste the Neon connection string from step 1.
   - `SECRET_KEY` — generate one: `python -c "import secrets; print(secrets.token_urlsafe(50))"`.
4. Click **Apply**. Render builds `Dockerfile.render` (~5-8 min expected;
   each failed build costs real build-minute quota, so double-check
   `DATABASE_URL`/`SECRET_KEY` are entered correctly before applying).

### 4. First-deploy verification
Watch the deploy logs for, in order:
1. `Applying users.0004_decision_reviewlabel... OK` (and the other
   migrations) — confirms `DATABASE_URL` is reachable.
2. `render_seed: ... seeding from scratch` then `render_seed: done. Scored
   N events in Xs` — **check X against the 45s budget**. If it consistently
   hits the cutoff, lower `RENDER_SEED_N_SCORE` (Render dashboard → service
   → Environment) to 100-150 and trigger a redeploy (env-var-only change,
   no rebuild).
3. `Sentinel scoring service ready. cold=... warm=... store=memory` — the
   FastAPI sub-app's lifespan ran.
4. Uvicorn's own "Application startup complete" / listening line.

Then open the assigned `https://<service>.onrender.com` URL: the landing
page should render, `/prediction/` (after registering an account) should
score a transaction, `/monitoring/` should show the seeded decisions, and
`/api/docs` should show the FastAPI Swagger UI.

### 5. Known free-tier behaviors, not bugs
- **Sleeps after 15 min idle.** The next request after a sleep takes
  noticeably longer (cold start: container boot + `render_seed.py`'s
  idempotent no-op check, which is fast since it just skips once >= 500
  rows exist — the slow first-ever seed only happens once, on the very
  first deploy).
- **The "Replay N events" button will be slow** — the monitoring page
  already says why (0.1 CPU vs. the 27ms p99 measured on dedicated local
  hardware).
