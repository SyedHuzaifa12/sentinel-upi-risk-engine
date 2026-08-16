"""Shared test fixtures. Deliberately hand-built UPIEvent instances, not the
full ml/src/generator pipeline -- feature_lib's tests stay independent of
generator internals and run fast.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from feature_lib.event import UPIEvent  # noqa: E402

_DEFAULTS = dict(
    txn_type="P2P",
    initiation_mode="INTENT",
    payer_bank="okaxis",
    payee_bank="ybl",
    payer_account_age_days=365,
    label_is_fraud=False,
    label_typology="legit",
)


def make_event(txn_id, timestamp, payer_vpa, payee_vpa, amount=500.0, device_id="dev-1", **overrides):
    fields = dict(_DEFAULTS)
    fields.update(overrides)
    return UPIEvent(
        txn_id=txn_id,
        timestamp=timestamp,
        payer_vpa=payer_vpa,
        payee_vpa=payee_vpa,
        amount=amount,
        device_id=device_id,
        **fields,
    )


# FLOAT_REL_TOL / FLOAT_ABS_TOL: InMemoryHistoryStore computes variance-based
# features (e.g. amount_zscore_vs_payer_30d) via an incremental running
# sum-of-squares in Python, while PostgresHistoryStore computes the same
# quantity via SQL's STDDEV_POP() aggregate -- a different implementation,
# summing in a different order. Both are correct; IEEE 754 doubles are not
# associative under addition, so two correct algorithms for the same formula
# can disagree in the last bit or two (observed: ~5e-16 relative difference
# on amount_zscore_vs_payer_30d against a live Postgres parity run). 1e-9 is
# ~7 orders of magnitude looser than that noise floor and ~7 orders of
# magnitude tighter than anything that could move a model's predicted
# probability across a decision threshold -- this is "ignore floating-point
# noise," not "accept a real numeric discrepancy." NaN-vs-NaN and missing-ness
# are still required to match EXACTLY (a value missing in one store and
# present in the other IS a real bug), and bools/ints/strings get no
# tolerance at all.
FLOAT_REL_TOL = 1e-9
FLOAT_ABS_TOL = 1e-12


def _rel_diff(a: float, b: float) -> float:
    if a == b:
        return 0.0
    denom = max(abs(a), abs(b), 1e-300)
    return abs(a - b) / denom


def assert_vectors_equal(values_a: dict, values_b: dict) -> float:
    """Exact equality for bools/ints/strings and for NaN-vs-NaN/missingness;
    a float-vs-float comparison tolerates only IEEE-754-noise-level
    disagreement (see FLOAT_REL_TOL/FLOAT_ABS_TOL above). Returns the maximum
    relative float difference observed, so callers can print it -- future
    drift beyond noise should be visible, not silently absorbed."""
    assert values_a.keys() == values_b.keys()

    mismatches = {}
    max_rel_diff = 0.0
    for key in values_a:
        a, b = values_a[key], values_b[key]
        if isinstance(a, float) and isinstance(b, float):
            a_nan, b_nan = (a != a), (b != b)
            if a_nan or b_nan:
                if a_nan != b_nan:  # exactly one is NaN -- never acceptable
                    mismatches[key] = (a, b)
                continue  # both NaN: matches
            if not math.isclose(a, b, rel_tol=FLOAT_REL_TOL, abs_tol=FLOAT_ABS_TOL):
                mismatches[key] = (a, b)
            else:
                max_rel_diff = max(max_rel_diff, _rel_diff(a, b))
        elif a != b:
            mismatches[key] = (a, b)

    assert not mismatches, f"feature vectors differ: {mismatches}"
    return max_rel_diff
