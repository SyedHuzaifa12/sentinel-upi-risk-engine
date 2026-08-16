"""FastAPI scoring service entry point.

    uvicorn service.main:app --port 8001

Startup order (in the lifespan below) is load-bearing, not incidental:
  1. scoring.load_models() -- which itself compares
     ml/artifacts/models/feature_registry.json against feature_lib.registry
     FIRST and refuses to load anything on a mismatch (see service/scoring.py;
     this guard moved there in Phase 5 so worker/consumer.py and
     worker/ingest.py get it for free too, instead of each needing their own
     copy).
  2. Build the HistoryStore (STORE_BACKEND env var).
  3. Optionally seed it from data/synthetic/events.jsonl (SEED_EVENTS env var).
A guard that ran after model loading would be too late -- a mismatched
model would already be resident and could already be serving requests
racing the startup check.
"""
import json
import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI  # noqa: E402

from feature_lib.registry import ALL_FEATURES, COLD_FEATURES  # noqa: E402
from ml.src.utils.paths import METRICS_V2_PATH, THRESHOLDS_PATH  # noqa: E402

from . import config, scoring  # noqa: E402
from .deps import build_store  # noqa: E402
from .routes import router  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("service")


def _thresholds_version():
    with open(THRESHOLDS_PATH, "rb") as f:
        import hashlib
        return hashlib.sha256(f.read()).hexdigest()[:8]


def _trained_at_utc():
    try:
        with open(METRICS_V2_PATH) as f:
            return json.load(f).get("trained_at_utc")
    except FileNotFoundError:
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    registry = scoring.load_models()
    logger.info(
        "Feature registry startup guard passed. Loaded models: cold=%s warm=%s",
        registry["cold_model"]["model_version"], registry["warm_model"]["model_version"],
    )

    app.state.store = build_store()
    app.state.store_backend = config.STORE_BACKEND
    logger.info("Store backend: %s", config.STORE_BACKEND)

    # Decision logging requires Postgres (DATABASE_URL). Falls back to None
    # (skip logging, not a crash) when unset -- this is what keeps
    # `uvicorn service.main:app` usable for local dev/tests without Docker,
    # exactly like STORE_BACKEND already defaults to "memory" for the same
    # reason. In docker-compose.yml, DATABASE_URL is always set, so every
    # decision IS logged there, per CLAUDE.md's "every decision is logged
    # immutably" rule -- this is an explicit dev-mode exception, not a
    # silent gap.
    if config.DATABASE_URL:
        from decisionlog import DecisionLog
        app.state.decision_log = DecisionLog(config.DATABASE_URL)
        logger.info("Decision log connected (Postgres).")
    else:
        app.state.decision_log = None
        logger.warning(
            "DATABASE_URL not set -- decision logging disabled. This is fine for local "
            "dev/tests; docker-compose.yml sets DATABASE_URL so every decision is logged there."
        )

    if config.SEED_EVENTS:
        from ml.src.data.load_synthetic import load_synthetic_events
        events = load_synthetic_events()
        for event in events:
            app.state.store.record(event)
        logger.info("Seeded store with %d historical events from data/synthetic/events.jsonl", len(events))

    app.state.logger = logger
    app.state.started_at_epoch = time.time()
    app.state.model_info = {
        "cold_model_version": registry["cold_model"]["model_version"],
        "warm_model_version": registry["warm_model"]["model_version"],
        "thresholds_version": _thresholds_version(),
        "cold_feature_count": len(COLD_FEATURES),
        "warm_feature_count": len(ALL_FEATURES),
        "trained_at_utc": _trained_at_utc(),
    }

    print(f"Sentinel scoring service ready. cold={registry['cold_model']['model_version']} "
          f"warm={registry['warm_model']['model_version']} store={config.STORE_BACKEND} "
          f"seeded={config.SEED_EVENTS}")

    yield


app = FastAPI(title="Sentinel UPI Risk Engine -- scoring service", lifespan=lifespan)
app.include_router(router)
