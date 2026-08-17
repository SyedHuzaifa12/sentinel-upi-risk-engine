# Render deployment readiness notes

Investigation only — **no code changes made for this**. Written ahead of a
planned slimmed single-container deploy to Render's free tier (512MB RAM,
0.1 CPU).

## 1. decisionlog on SQLite

**Not viable as-is — and the blocker is deeper than the trigger SQL.**

`decisionlog/writer.py` hardcodes `import psycopg` (psycopg3) and connects
via `psycopg.connect(dsn, autocommit=True)`. psycopg speaks the Postgres
wire protocol only — it cannot open a SQLite file at all, regardless of
schema. So this isn't "the append-only guarantee silently stops working on
SQLite," it's "the process crashes on `DecisionLog.__init__` before it ever
gets that far."

Separately, `decisionlog/schema.py`'s append-only enforcement is also
Postgres-specific SQL:

```sql
CREATE OR REPLACE FUNCTION decisions_prevent_mutation() RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'decisions is append-only: % is not allowed', TG_OP;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER decisions_no_update
BEFORE UPDATE ON decisions
FOR EACH ROW EXECUTE FUNCTION decisions_prevent_mutation();
```

`LANGUAGE plpgsql` and `CREATE OR REPLACE TRIGGER ... EXECUTE FUNCTION`
don't exist in SQLite. SQLite *does* have triggers, just a different
dialect — the equivalent would be:

```sql
CREATE TRIGGER decisions_no_update BEFORE UPDATE ON decisions
BEGIN SELECT RAISE(ABORT, 'decisions is append-only: UPDATE is not allowed'); END;
```

**What an app-level guard would actually need**, if SQLite is the real
target: a parallel `DecisionLog` implementation using Python's stdlib
`sqlite3` (not psycopg) plus a SQLite-dialect `schema.py` (the `RAISE(ABORT,
...)` form above) — a genuine, if small, second backend, not a config flag.
`decisionlog/tests` would also need a SQLite variant of every test, since
the current suite skips entirely without `TEST_DATABASE_URL` pointed at a
real Postgres.

**Recommendation:** don't build that second backend. Render offers a free
Postgres tier — use it, and keep `decisionlog` exactly as it is. The
SQLite path is real work for no benefit here; the whole point of
`decisionlog` is DB-enforced immutability, which is easiest to keep exactly
where it's already proven correct.

## 2. FastAPI mounted inside Django's process

**Not a hard blocker — Django can run as ASGI, no new code needed there.**

`backend/config/asgi.py` already exists (stock
`django.core.asgi.get_asgi_application()`) — Django doesn't need to be
rewritten to speak ASGI, it already can. The composition path is: keep
Django as WSGI (simpler, nothing about its own code needs `async def`),
wrap it in `starlette.middleware.wsgi.WSGIMiddleware` (confirmed importable
in this venv, `starlette==0.41.3` — FastAPI depends on Starlette already,
so this is not a new dependency), and mount both apps under one Starlette
router, served by one `uvicorn`:

```python
from starlette.applications import Starlette
from starlette.middleware.wsgi import WSGIMiddleware
from starlette.routing import Mount

from backend.config.wsgi import application as django_wsgi_app
from service.main import app as fastapi_app

app = Starlette(routes=[
    Mount("/api", app=fastapi_app),
    Mount("/", app=WSGIMiddleware(django_wsgi_app)),
])
```

**Real open question, not yet verified**: whether Starlette's `Mount`
propagates ASGI `lifespan` events down into the mounted FastAPI sub-app.
`service/scoring.load_models()` runs during FastAPI's own `lifespan` context
today (`service/main.py`) — if `Mount` doesn't forward `lifespan.startup` to
the sub-app, `load_models()` would never run and every `/api/v1/score` call
would 500. This needs a direct test (start the combined app, hit `/api/v1/
score` once) before treating the composition as done, not just assumed
from Starlette's docs.

Also not yet checked: whether `worker/consumer.py`'s Streams consumer loop
can run inside the same single process (as a background `asyncio.create_task`
at FastAPI startup) or needs to stay a separate process — Render's free tier
is typically one web service, no separate background worker dyno, so this
matters for whether replay/live scoring works at all on that tier, not just
for the API path.

## 3. Estimated single-process RSS

Live-measured just now (`docker stats`, this repo's current containers, not
the user-supplied figures from the original ask — those were slightly
stale):

| Container | MEM USAGE |
|---|---|
| api | 206.8 MiB |
| ui | 144.1 MiB |
| worker | 201.5 MiB |

**Correction to the "duplicated numpy/pandas" assumption**: grepped
`backend/` for direct imports — `ui`'s Django process imports `numpy` only
(for ULID generation in `views/prediction.py`), and imports **none** of
pandas/lightgbm/scikit-learn/shap directly. It's an HTTP client of the
FastAPI service (Phase 4 decision), not a model host. So `ui`'s ~144MB is
Django + its own dependency stack (psycopg2, social-auth, whitenoise,
cryptography, dj-database-url) — **not** a second copy of the ML libraries.
`api`'s ~207MB carries essentially all of that weight already (numpy,
pandas, scikit-learn, lightgbm, shap, fastapi, uvicorn).

Merging the two into one process would **not** deduplicate an ML-library
copy that doesn't currently exist twice. The real savings are one Python
interpreter instead of two, and one supervisor process (gunicorn+uvicorn)
instead of two — realistically **~30-50MB**, not "half of 207MB."

**Rough combined estimate: ~260-320MB** (api's ~207MB is essentially fixed;
ui's Django-specific overhead after subtracting a redundant Python
interpreter base is maybe ~90-110MB on top). That leaves roughly
**190-250MB of headroom under Render's 512MB ceiling** — measured
containers, reasoned combination, not a load test.

**The tighter constraint is very likely CPU, not RAM.** Render's free tier
gives 0.1 vCPU (10% of one core). SHAP's `TreeExplainer` is the dominant
per-request cost measured in this project (Phase 3: p99≈20.9ms for
`compute_reason_codes` alone, out of a ~27ms total p99 budget on *dedicated*
local hardware). Under a 10%-of-one-core throttle, that same computation
could take an order of magnitude longer under any concurrent load — this
is a real risk to validate with an actual load test on Render before
trusting the RAM estimate as the binding constraint.

## Summary

| Question | Verdict |
|---|---|
| decisionlog on SQLite | Blocked at the connection layer (psycopg-only), not just the trigger SQL. Use Render's free Postgres instead of building a second backend. |
| FastAPI + Django, one process | Not blocked — Django already has `asgi.py`; compose via `WSGIMiddleware` + Starlette `Mount`. Verify `lifespan` propagation and worker placement before relying on it. |
| Single-process RAM | ~260-320MB estimated, real headroom under 512MB — but CPU (0.1 vCPU) is the more likely actual constraint given SHAP's measured cost, not RAM. |
