"""The decision policy: turns a calibrated probability into an action.

Pure function, no I/O per call -- thresholds and model versions are loaded
once at module import time (same eager-load pattern as
backend/users/services/prediction_service.py).
"""
import hashlib
import json
from dataclasses import dataclass

from ml.src.utils.paths import FEATURE_REGISTRY_PATH, THRESHOLDS_PATH

RISK_TIERS = ("LOW", "MEDIUM", "HIGH", "CRITICAL")
ACTIONS = ("ALLOW", "WARN", "REVIEW", "BLOCK")


@dataclass
class Decision:
    action: str            # ALLOW | WARN | REVIEW | BLOCK
    risk_score: float       # the calibrated probability, unchanged
    risk_tier: str           # LOW | MEDIUM | HIGH | CRITICAL
    thresholds_version: str
    model_version: str


def _load_thresholds():
    with open(THRESHOLDS_PATH) as f:
        data = json.load(f)
    # Version = content hash of the thresholds file itself, so it changes
    # whenever thresholds are re-optimized, without needing a separate
    # manually-bumped version number.
    with open(THRESHOLDS_PATH, "rb") as f:
        version = hashlib.sha256(f.read()).hexdigest()[:8]
    return data, version


def _load_model_versions():
    with open(FEATURE_REGISTRY_PATH) as f:
        registry = json.load(f)
    return {
        "cold": registry["cold_model"]["model_version"],
        "warm": registry["warm_model"]["model_version"],
    }


_THRESHOLDS, _THRESHOLDS_VERSION = _load_thresholds()
_MODEL_VERSIONS = _load_model_versions()


def _risk_tier(prob, thresholds):
    if prob < thresholds["t_warn"]:
        return "LOW"
    if prob < thresholds["t_review"]:
        return "MEDIUM"
    if prob < thresholds["t_block"]:
        return "HIGH"
    return "CRITICAL"


def _action_for_tier(tier):
    return {"LOW": "ALLOW", "MEDIUM": "WARN", "HIGH": "REVIEW", "CRITICAL": "BLOCK"}[tier]


def decide(calibrated_prob: float, is_cold: bool, amount: float) -> Decision:
    """Pure: same inputs always produce the same Decision. `amount` isn't
    used in the tier/action logic itself (that's threshold-on-probability
    only) -- it's accepted here because a real deployment's cost-aware
    policy may want to escalate high-value borderline cases, and keeping
    it in the signature now avoids a breaking change later; it is not yet
    used for that in this phase.
    """
    model_key = "cold" if is_cold else "warm"
    thresholds = _THRESHOLDS[model_key]

    tier = _risk_tier(calibrated_prob, thresholds)
    action = _action_for_tier(tier)

    return Decision(
        action=action,
        risk_score=calibrated_prob,
        risk_tier=tier,
        thresholds_version=_THRESHOLDS_VERSION,
        model_version=_MODEL_VERSIONS[model_key],
    )
