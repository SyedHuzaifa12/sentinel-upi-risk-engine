"""Per-transaction reason codes: replaces the hardcoded HTML explanation
panels. Uses SHAP TreeExplainer on the actual cold/warm LightGBM models, so
every reason code is generated from that specific transaction's real
feature values -- never a canned sentence.

Import-order note (load-bearing, do not reorder): on this platform, if
`pandas` is imported before the bare `lightgbm` package, LightGBM's ctypes
bridge crashes with a native access-violation the first time its C API is
touched (verified directly by bisecting the import order -- reproducible
regardless of whether `shap` is involved at all). `import lightgbm` must be
the first import in this module, before `pandas`.
"""
import math
import sys
import time
from pathlib import Path

import joblib
import lightgbm  # noqa: F401  -- MUST be imported before pandas, see module docstring
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from feature_lib.registry import ALL_FEATURES, COLD_FEATURES, REGISTRY  # noqa: E402
from ml.src.utils.paths import COLD_MODEL_PATH, WARM_MODEL_PATH  # noqa: E402

_cold_model = joblib.load(COLD_MODEL_PATH)
_warm_model = joblib.load(WARM_MODEL_PATH)

# Lazy-loaded (2026-08-17, Render deploy target): shap.TreeExplainer's own
# construction is real, measurable startup cost -- on a slow enough CPU
# (e.g. Render free tier's 0.1 vCPU) it's worth deferring past process
# startup, not just past model loading. Built once, on the first ACTUAL
# non-ALLOW decision, and cached from then on -- every call after the first
# is exactly as fast as the eager version was. This intentionally reverses
# service/scoring.py's Phase 3 docstring claim that explainer construction
# happens "at IMPORT time" -- see that module's own updated comment.
_cold_explainer = None
_warm_explainer = None


def _get_explainer(is_cold: bool):
    global _cold_explainer, _warm_explainer
    if is_cold:
        if _cold_explainer is None:
            _cold_explainer = shap.TreeExplainer(_cold_model)
        return _cold_explainer
    if _warm_explainer is None:
        _warm_explainer = shap.TreeExplainer(_warm_model)
    return _warm_explainer


CATEGORICAL_FEATURES = ["amount_roundness"]
LATENCY_BUDGET_MS = 20.0


def _pct(v):
    return f"{v * 100:.0f}%"


_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# One template per BASE feature (37). _is_missing companions (14) are
# derived mechanically below -- never hand-duplicated, so there's nothing
# to keep in sync when a feature is added/renamed in feature_lib.
_BASE_TEMPLATES = {
    "amount_log": lambda v: f"Transaction amount is on the higher end for this kind of payment (log-amount {v:.2f})",
    "hour_of_day": lambda v: f"Transaction occurred at {int(v):02d}:00",
    "is_night": lambda v: ("Payment made outside this payer's usual hours" if v
                            else "Payment made during this payer's usual hours"),
    "day_of_week": lambda v: f"Transaction occurred on {_WEEKDAYS[int(v) % 7]}",
    "is_collect_request": lambda v: ("Initiated as a collect request" if v
                                      else "Not initiated as a collect request"),
    "is_p2p": lambda v: "Person-to-person transfer" if v else "Person-to-merchant payment",
    "amount_roundness": lambda v: f"Amount pattern: {v}",
    "payer_account_age_days": lambda v: f"Payer's account is {v:.0f} days old",
    "amount_zscore_vs_payer_30d": lambda v: f"Amount is {v:.1f} standard deviations from this payer's 30-day average",
    "amount_ratio_to_payer_median": lambda v: f"Amount is {v:.0f}x this payer's typical payment",
    "payer_txn_count_1h": lambda v: f"{v:.0f} payments from this payer in the last hour",
    "payer_txn_count_24h": lambda v: f"{v:.0f} payments from this payer in the last 24 hours",
    "payer_txn_count_30d": lambda v: f"{v:.0f} payments from this payer in the last 30 days",
    "payer_distinct_payees_24h": lambda v: f"Paid {v:.0f} different payees in the last 24 hours",
    "seconds_since_payer_last_txn": lambda v: f"{v / 60:.0f} minutes since this payer's last payment",
    "hour_deviation_from_payer_normal": lambda v: f"Transaction hour is {v:.1f} hours from this payer's normal pattern",
    "payer_new_payee_rate_30d": lambda v: f"This payer pays new payees {_pct(v)} of the time",
    "payer_collect_request_rate_30d": lambda v: f"This payer uses collect requests {_pct(v)} of the time",
    "is_first_ever_txn_by_payer": lambda v: ("This is this payer's first-ever transaction" if v
                                              else "This payer has transacted before"),
    "payee_age_hours_in_system": lambda v: f"Payee first seen {v:.0f} hours ago",
    "payee_total_txn_count": lambda v: f"Payee has {v:.0f} total transactions on record",
    "payee_is_unseen": lambda v: ("This payee has never been seen before" if v
                                   else "This payee has a transaction history"),
    "payee_handle_digit_ratio": lambda v: f"Payee handle is {_pct(v)} digits",
    "payee_handle_entropy": lambda v: f"Payee handle has {v:.1f} bits of character randomness",
    "payee_handle_has_dictionary_name": lambda v: ("Payee handle contains a recognizable name" if v
                                                    else "Payee handle does not contain a recognizable name"),
    "payee_distinct_payers_1h": lambda v: f"{v:.0f} different payers sent money to this payee in the last hour",
    "payee_distinct_payers_24h": lambda v: f"{v:.0f} different payers sent money to this payee in the last 24 hours",
    "payee_distinct_payers_since_first_seen": lambda v: f"{v:.0f} different payers have ever paid this payee",
    "payee_fanin_velocity": lambda v: f"This payee is receiving from {v:.1f} distinct payers per hour",
    "payee_amount_mean": lambda v: f"This payee's average received amount is Rs {v:.0f}",
    "payee_amount_std": lambda v: f"This payee's received amounts vary by Rs {v:.0f} (std dev)",
    "payee_p2m_ratio": lambda v: f"{_pct(v)} of this payee's transactions are merchant-style payments",
    "payer_payee_txn_count": lambda v: ("First payment ever between this payer and payee" if v == 0
                                        else f"{v:.0f} prior payments between this payer and payee"),
    "days_since_payer_last_paid_payee": lambda v: f"{v:.0f} days since this payer last paid this payee",
    "device_distinct_payers_30d": lambda v: f"{v:.0f} different payers have used this device in the last 30 days",
    "device_txn_count_1h": lambda v: f"{v:.0f} transactions from this device in the last hour",
    "is_new_device_for_payer": lambda v: ("This device has not been used by this payer before" if v
                                          else "This device has been used by this payer before"),
}


def _missing_template(base_name):
    display = base_name.replace("_", " ")
    return lambda v: f"{display.capitalize()} could not be reliably computed (insufficient history)"


def _build_full_template_registry():
    templates = {}
    for spec in REGISTRY:
        if spec.name in _BASE_TEMPLATES:
            templates[spec.name] = _BASE_TEMPLATES[spec.name]
        elif spec.name.endswith("_is_missing"):
            base_name = spec.name[: -len("_is_missing")]
            templates[spec.name] = _missing_template(base_name)
        else:
            raise KeyError(f"No reason-code template for registry feature '{spec.name}'")
    return templates


TEMPLATES = _build_full_template_registry()


def _prepare_row(feature_vector: dict, feature_columns: list):
    row = {name: feature_vector.get(name) for name in feature_columns}
    df = pd.DataFrame([row], columns=feature_columns)
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def compute_reason_codes(feature_vector: dict, is_cold: bool, top_n: int = 5) -> list:
    """Returns up to `top_n` reason codes for the features pushing the
    score UP the most, ordered by absolute SHAP contribution descending.
    Each entry: {code, message, feature, value, shap_contribution}.
    `message` never contains a raw feature name.
    """
    explainer = _get_explainer(is_cold)
    feature_columns = COLD_FEATURES if is_cold else ALL_FEATURES

    X = _prepare_row(feature_vector, feature_columns)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]  # older shap: [neg_class, pos_class] -- take positive class
    contributions = shap_values[0]

    ranked = sorted(
        ((name, contributions[i]) for i, name in enumerate(feature_columns)),
        key=lambda pair: abs(pair[1]),
        reverse=True,
    )
    positive = [(name, contrib) for name, contrib in ranked if contrib > 0]

    codes = []
    for name, contrib in positive[:top_n]:
        value = feature_vector.get(name)
        # A base feature can be NaN (insufficient history to compute it) yet
        # still be the top SHAP contributor -- LightGBM routes NaN through a
        # learned split, so it's a real, rankable signal, not noise. Its own
        # template (e.g. "{v:.0f} days since...") would render the literal
        # string "nan" in that case; fall back to the same "could not be
        # reliably computed" wording its _is_missing companion already uses,
        # rather than ever putting "nan" in front of an analyst.
        if isinstance(value, float) and math.isnan(value):
            message = _missing_template(name)(value)
        else:
            message = TEMPLATES[name](value)
        codes.append({
            "code": name,
            "message": message,
            "feature": name,
            "value": value,
            "shap_contribution": float(contrib),
        })
    return codes


def measure_shap_latency_ms(feature_vectors: list, is_cold_flags: list) -> dict:
    """Wall-clock per-call SHAP time over a batch of real rows. Returns
    {p50, p99, n} in milliseconds. Never used to decide whether to drop
    reason codes -- only to report the measured cost honestly."""
    durations = []
    for vector, is_cold in zip(feature_vectors, is_cold_flags):
        start = time.perf_counter()
        compute_reason_codes(vector, is_cold)
        durations.append((time.perf_counter() - start) * 1000)

    durations.sort()
    n = len(durations)

    def _pctl(p):
        idx = min(n - 1, int(round(p * (n - 1))))
        return durations[idx]

    return {"p50_ms": _pctl(0.50), "p99_ms": _pctl(0.99), "n": n}
