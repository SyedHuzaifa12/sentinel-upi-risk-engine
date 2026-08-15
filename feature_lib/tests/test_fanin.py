"""Fan-in: 40 distinct payers to one new VPA in 1h produces high
payee_fanin_velocity, without the floor fix exploding it."""
from datetime import datetime, timedelta, timezone

from feature_lib.compute import compute_features, compute_features_batch
from feature_lib.store.in_memory import InMemoryHistoryStore

from .conftest import make_event

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_forty_distinct_payers_in_one_hour_produces_high_fanin():
    store = InMemoryHistoryStore()

    mule_vpa = "xyz9921@ybl"
    events = [
        make_event(f"t{i}", T0 + timedelta(minutes=i), f"payer{i}@x", mule_vpa, amount=300.0)
        for i in range(40)
    ]
    compute_features_batch(events, store)

    # 41st event to the same payee, just after the fan-in window.
    target = make_event("t-target", T0 + timedelta(minutes=45), "payer-41@x", mule_vpa, amount=300.0)
    vector = compute_features(target, store)

    assert vector.values["payee_distinct_payers_1h"] == 40
    assert vector.values["payee_distinct_payers_since_first_seen"] == 40

    # Floored at 1.0h even though only ~45 minutes have elapsed since first
    # seen -- so velocity should sit close to 40/hour, not explode past it
    # the way an unfloored (elapsed_hours < 1) denominator would.
    velocity = vector.values["payee_fanin_velocity"]
    assert 30 <= velocity <= 45, velocity
