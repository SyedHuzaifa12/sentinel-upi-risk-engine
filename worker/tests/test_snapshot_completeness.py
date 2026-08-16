"""decisions.feature_snapshot is the basis of audit and Phase 6 replay -- if
it silently stored a subset, backtesting would be broken without anything
surfacing it. This pins down both the shape (exactly ALL_FEATURES, no
missing, no extras) and the actual replay guarantee (recomputing against an
identically-prepared store reproduces the same risk_score)."""
import os
import uuid
from datetime import datetime, timedelta, timezone

from feature_lib.registry import ALL_FEATURES
from feature_lib.store.redis_store import RedisHistoryStore
from feature_lib.tests.conftest import assert_vectors_equal
from service import scoring

from .conftest import make_event


def test_feature_snapshot_is_exactly_all_features_and_replay_reproduces_the_score():
    redis_url = os.environ.get("TEST_REDIS_URL")
    if not redis_url:
        import pytest
        pytest.skip("TEST_REDIS_URL not set -- no Redis reachable in this environment")

    store_original = RedisHistoryStore(redis_url, prefix=f"test_snapshot_orig_{uuid.uuid4().hex[:8]}", flush=True)
    store_replay = RedisHistoryStore(redis_url, prefix=f"test_snapshot_replay_{uuid.uuid4().hex[:8]}", flush=True)
    try:
        t0 = datetime.now(timezone.utc) - timedelta(days=2)
        payee = f"snapshot-payee-{uuid.uuid4().hex[:8]}@ybl"
        for i in range(4):
            setup_event = make_event(txn_id=f"snapshot-setup-{i}", payee_vpa=payee, timestamp=t0 + timedelta(hours=i))
            store_original.record(setup_event)
            store_replay.record(setup_event)

        target_event = make_event(
            txn_id=f"snapshot-target-{uuid.uuid4().hex[:8]}", payee_vpa=payee, timestamp=t0 + timedelta(days=1))

        original = scoring.score_event(target_event, store_original)

        # Completeness: every feature_lib registry feature is present, nothing missing, nothing extra.
        assert set(original.feature_snapshot.keys()) == set(ALL_FEATURES)

        # Replay guarantee: an identically-prepared store, given the same
        # event, reproduces the same risk_score -- this is what makes the
        # logged feature_snapshot + event a faithful, replayable record.
        replay_event = target_event.model_copy()
        replayed = scoring.score_event(replay_event, store_replay)

        assert round(original.risk_score, 6) == round(replayed.risk_score, 6)
        # NaN-aware comparison -- see test_shared_path.py's comment on why a
        # plain `==` is wrong here.
        assert_vectors_equal(original.feature_snapshot, replayed.feature_snapshot)
    finally:
        store_original.close()
        store_replay.close()
