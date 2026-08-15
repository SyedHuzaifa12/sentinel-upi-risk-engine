"""Shared test fixtures. Deliberately hand-built UPIEvent instances, not the
full ml/src/generator pipeline -- feature_lib's tests stay independent of
generator internals and run fast.
"""
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


def _values_equal(a, b):
    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        return True  # both NaN
    return a == b


def assert_vectors_equal(values_a: dict, values_b: dict):
    assert values_a.keys() == values_b.keys()
    mismatches = {
        key: (values_a[key], values_b[key])
        for key in values_a
        if not _values_equal(values_a[key], values_b[key])
    }
    assert not mismatches, f"feature vectors differ: {mismatches}"
