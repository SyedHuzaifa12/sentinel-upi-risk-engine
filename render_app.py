"""Render deploy target: composes Django (WSGI) and the FastAPI scoring
service (ASGI) into ONE process behind ONE port, for a slimmed single-
container free-tier deploy (512MB RAM, 0.1 CPU).

    uvicorn render_app:app --host 0.0.0.0 --port $PORT

This is an ADDED second deploy target, not a replacement -- docker-compose.yml
/ Dockerfile / worker/ / pipelines/ / MLflow are untouched and remain the
primary local-dev target. See DEPLOY_NOTES.md.

Routes: "/" -> Django (WSGI, wrapped), "/api" -> the FastAPI scoring service
(same app as `uvicorn service.main:app` in docker-compose's `api` service).

Import-order note (load-bearing, same issue as pipelines/backtest.py,
pipelines/drift_check.py, ml/src/policy/reason_codes.py, service/scoring.py):
`import lightgbm` must be the literal first import in this process, before
anything that transitively imports pandas.

Lifespan note (load-bearing, verified empirically before writing this --
see DEPLOY_NOTES.md "FastAPI mounted inside Django's process"): Starlette's
`Mount()` does NOT forward ASGI lifespan startup/shutdown events into a
mounted sub-app. `service.main.app`'s own `lifespan=` would simply never
fire if this file just did `Mount("/api", app=fastapi_app)` and stopped
there -- `scoring.load_models()` would never run, and every `/api/v1/score`
call would raise "load_models() must run before score_event()". The fix:
THIS app's own `lifespan=` explicitly delegates to the existing, already-
correct `service.main.lifespan` function, called against the `fastapi_app`
instance -- zero duplicated startup logic, and confirmed working (request
handlers inside a mounted sub-app see `request.app is <that sub-app>`, so
state set on `fastapi_app` by the delegated lifespan is visible to
`service/routes.py`'s `request.app.state.*` reads).
"""
import lightgbm  # noqa: F401,E402,I001 -- MUST be imported before pandas, see module docstring
import sys
from contextlib import asynccontextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = REPO_ROOT / "backend"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(BACKEND_DIR))  # so `import config.wsgi` (Django's project package) resolves

# Importing config.wsgi triggers django.setup() -- must happen after
# BACKEND_DIR is on sys.path (above), and works whether or not
# DJANGO_SETTINGS_MODULE was already set by the entrypoint script.
from config.wsgi import application as django_wsgi_app  # noqa: E402
from starlette.applications import Starlette  # noqa: E402
from starlette.middleware.wsgi import WSGIMiddleware  # noqa: E402
from starlette.routing import Mount  # noqa: E402

from service.main import app as fastapi_app  # noqa: E402
from service.main import lifespan as fastapi_lifespan  # noqa: E402


@asynccontextmanager
async def combined_lifespan(_app):
    async with fastapi_lifespan(fastapi_app):
        yield


app = Starlette(
    routes=[
        Mount("/api", app=fastapi_app),
        Mount("/", app=WSGIMiddleware(django_wsgi_app)),
    ],
    lifespan=combined_lifespan,
)
