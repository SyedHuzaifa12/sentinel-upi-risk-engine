"""The one scoring path: features -> route cold/warm -> predict -> calibrate
-> decide -> reason codes (non-ALLOW only).

Models/calibrators are NOT loaded at import time -- call load_models() once,
from service/main.py's lifespan, AFTER the feature-registry startup guard
has verified the saved models agree with feature_lib.registry. That
ordering is load-bearing: it's what lets the guard actually refuse to start
before anything (possibly stale/mismatched) is resident in memory. Do not
move the joblib.load calls back to module level.

Import-order note (load-bearing, do not reorder -- same issue documented in
ml/src/policy/reason_codes.py): if `pandas` is imported before the bare
`lightgbm` package anywhere in this process, LightGBM's ctypes bridge
crashes with a native access-violation the first time its C API is touched.
`import lightgbm` must happen before `feature_lib.frame` (which imports
pandas) is imported anywhere below.
"""
import lightgbm  # noqa: F401 -- MUST be imported before pandas, see module docstring
import time
from dataclasses import dataclass, field

import joblib
import numpy as np

from feature_lib.compute import compute_features
from feature_lib.event import UPIEvent
from feature_lib.frame import vector_to_frame
from feature_lib.registry import ALL_FEATURES, COLD_FEATURES
from feature_lib.store.base import HistoryStore
from ml.src.policy.decide import decide
from ml.src.training.lightgbm_pipeline import apply_calibrator
from ml.src.utils.paths import COLD_CALIBRATOR_PATH, COLD_MODEL_PATH, WARM_CALIBRATOR_PATH, WARM_MODEL_PATH

_cold_model = None
_warm_model = None
_cold_calibrator = None
_warm_calibrator = None
_compute_reason_codes = None
_loaded = False


def load_models():
    """Loads cold/warm models + calibrators, then -- only now -- imports
    ml.src.policy.reason_codes. That module eagerly loads its own model
    copies and builds SHAP TreeExplainers at IMPORT time (a Phase 3 design
    choice we're not changing); importing it lazily here rather than at
    this module's top level is what keeps that load from happening before
    the caller's startup guard has run.
    """
    global _cold_model, _warm_model, _cold_calibrator, _warm_calibrator, _compute_reason_codes, _loaded

    _cold_model = joblib.load(COLD_MODEL_PATH)
    _warm_model = joblib.load(WARM_MODEL_PATH)
    _cold_calibrator = joblib.load(COLD_CALIBRATOR_PATH)
    _warm_calibrator = joblib.load(WARM_CALIBRATOR_PATH)

    from ml.src.policy.reason_codes import compute_reason_codes
    _compute_reason_codes = compute_reason_codes
    _loaded = True


def is_loaded() -> bool:
    return _loaded


@dataclass
class ScoreResult:
    txn_id: str
    risk_score: float
    action: str
    risk_tier: str
    reason_codes: list
    model_version: str
    thresholds_version: str
    is_cold: bool
    features_computed: int
    latency_ms: float
    # Per-stage breakdown -- logged and read by service/benchmark.py, not
    # part of the /v1/score wire response (that returns latency_ms only).
    stage_latency_ms: dict = field(default_factory=dict)


def score_event(event: UPIEvent, store: HistoryStore) -> ScoreResult:
    if not _loaded:
        raise RuntimeError(
            "scoring.load_models() must run before score_event() -- see service/main.py's lifespan."
        )

    t0 = time.perf_counter()
    vector = compute_features(event, store)
    t1 = time.perf_counter()

    cold = vector.is_cold
    if cold:
        model, calibrator, feature_columns = _cold_model, _cold_calibrator, COLD_FEATURES
    else:
        model, calibrator, feature_columns = _warm_model, _warm_calibrator, ALL_FEATURES

    X = vector_to_frame([vector.values], feature_columns)
    proba_raw = model.predict_proba(X)[:, 1][0]
    t2 = time.perf_counter()

    proba_cal = float(apply_calibrator(calibrator, np.array([proba_raw]))[0])
    t3 = time.perf_counter()

    decision = decide(proba_cal, cold, event.amount)
    t4 = time.perf_counter()

    # Reason codes only for non-ALLOW actions: SHAP is the dominant latency
    # cost (Phase 3 measured p99=20.9ms for compute_reason_codes ALONE,
    # against this service's 100ms total p99 budget), and ALLOW is both the
    # overwhelming majority action and one with nothing adverse to explain.
    # See service/benchmark.py and PROGRESS.md's Phase 4 latency table for
    # the measurement backing this decision.
    reason_codes = [] if decision.action == "ALLOW" else _compute_reason_codes(vector.values, cold)
    t5 = time.perf_counter()

    # AFTER features are computed -- never before (feature_lib.compute's own contract).
    store.record(event)

    stage_ms = {
        "features_ms": (t1 - t0) * 1000,
        "predict_ms": (t2 - t1) * 1000,
        "calibrate_ms": (t3 - t2) * 1000,
        "decide_ms": (t4 - t3) * 1000,
        "reason_codes_ms": (t5 - t4) * 1000,
    }

    return ScoreResult(
        txn_id=event.txn_id,
        risk_score=decision.risk_score,
        action=decision.action,
        risk_tier=decision.risk_tier,
        reason_codes=reason_codes,
        model_version=decision.model_version,
        thresholds_version=decision.thresholds_version,
        is_cold=cold,
        features_computed=len(vector.values),
        latency_ms=(t5 - t0) * 1000,
        stage_latency_ms=stage_ms,
    )
