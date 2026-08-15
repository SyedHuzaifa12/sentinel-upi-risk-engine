"""PIT correctness: features for an event at time t must use ONLY rows with
timestamp < t. Proven behaviorally here (recompute after inserting a future
row, assert nothing changed), rather than by introspecting store internals.
"""
from datetime import datetime, timedelta, timezone

from feature_lib.compute import compute_features
from feature_lib.store.in_memory import InMemoryHistoryStore

from .conftest import assert_vectors_equal, make_event

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_future_event_does_not_change_a_past_events_features():
    store = InMemoryHistoryStore()

    past = make_event("t-past", T0, "payer@x", "payee@y", amount=100.0)
    store.record(past)

    target = make_event("t-target", T0 + timedelta(hours=1), "payer@x", "payee@y", amount=150.0)
    vector_before = compute_features(target, store)

    # A future event for the SAME payer.
    future_payer_event = make_event("t-future-payer", T0 + timedelta(hours=2), "payer@x", "someone-else@z",
                                     amount=999.0)
    store.record(future_payer_event)

    # A future event for the SAME payee.
    future_payee_event = make_event("t-future-payee", T0 + timedelta(hours=3), "another-payer@w", "payee@y",
                                     amount=888.0)
    store.record(future_payee_event)

    vector_after = compute_features(target, store)

    assert_vectors_equal(vector_before.values, vector_after.values)
    assert vector_before.is_cold == vector_after.is_cold
