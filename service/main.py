"""FastAPI scoring service entry point.

    uvicorn service.main:app --port 8001

Startup order (in the lifespan below) is load-bearing, not incidental:
  1. Compare ml/artifacts/models/feature_registry.json against
     feature_lib.registry -- refuse to start on any mismatch.
  2. Only once that passes: scoring.load_models() actually loads the
     cold/warm models + calibrators (+ SHAP explainers, via
     ml.src.policy.reason_codes) into memory.
  3. Build the HistoryStore (STORE_BACKEND env var).
  4. Optionally seed it from data/synthetic/events.jsonl (SEED_EVENTS env var).
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
from ml.src.utils.paths import FEATURE_REGISTRY_PATH, METRICS_V2_PATH, THRESHOLDS_PATH  # noqa: E402

from . import config, scoring  # noqa: E402
from .deps import build_store  # noqa: E402
from .routes import router  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("service")


def _assert_feature_registry_matches_code():
    with open(FEATURE_REGISTRY_PATH) as f:
        registry = json.load(f)

    mismatches = []
    if registry["cold_model"]["features"] != COLD_FEATURES:
        mismatches.append("cold_model.features != feature_lib.registry.COLD_FEATURES")
    if registry["warm_model"]["features"] != ALL_FEATURES:
        mismatches.append("warm_model.features != feature_lib.registry.ALL_FEATURES")

    if mismatches:
        raise RuntimeError(
            "Refusing to start: feature_registry.json disagrees with feature_lib.registry "
            f"({'; '.join(mismatches)}). The saved model was trained against a different "
            "feature set than the code currently computes -- retrain (python ml/src/train.py) "
            "before serving."
        )
    return registry


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
    registry = _assert_feature_registry_matches_code()
    logger.info("Feature registry startup guard passed.")

    scoring.load_models()
    logger.info(
        "Loaded models: cold=%s warm=%s",
        registry["cold_model"]["model_version"], registry["warm_model"]["model_version"],
    )

    app.state.store = build_store()
    app.state.store_backend = config.STORE_BACKEND
    logger.info("Store backend: %s", config.STORE_BACKEND)

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
