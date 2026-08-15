"""Cold path: a never-seen payee yields no usable group-D features and
is_cold() is True."""
import math
from datetime import datetime, timezone

from feature_lib.compute import compute_features, is_cold
from feature_lib.registry import BASE_REGISTRY
from feature_lib.store.in_memory import InMemoryHistoryStore

from .conftest import make_event

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_never_seen_payee_is_cold_and_group_d_is_all_missing():
    store = InMemoryHistoryStore()
    event = make_event("t1", T0, "payer@x", "brand-new-payee@y", amount=250.0)

    assert is_cold(event, store) is True

    vector = compute_features(event, store)
    assert vector.is_cold is True

    for spec in BASE_REGISTRY:
        if spec.group != "D":
            continue
        value = vector.values[spec.name]
        if spec.can_be_missing:
            assert vector.values[f"{spec.name}_is_missing"] is True, spec.name
            assert isinstance(value, float) and math.isnan(value), spec.name
        else:
            # payer_payee_txn_count: spec-mandated 0, not missing.
            assert value == 0, spec.name
