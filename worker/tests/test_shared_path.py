"""The important test: the API path (a direct scoring.score_event call) and
the worker path (worker.consumer._score_and_log, which wraps the SAME
function) must produce an identical score for the same event -- proving
there is one scoring implementation, not two."""
import os
import uuid
from datetime import datetime, timedelta, timezone

from feature_lib.store.redis_store import RedisHistoryStore
from feature_lib.tests.conftest import assert_vectors_equal
from service import scoring
from worker.consumer import _score_and_log
from worker.streams import encode_event

from .conftest import make_event


def test_api_and_worker_paths_produce_identical_score(decision_log):
    redis_url = os.environ["TEST_REDIS_URL"]
    store_a = RedisHistoryStore(redis_url, prefix=f"test_shared_a_{uuid.uuid4().hex[:8]}", flush=True)
    store_b = RedisHistoryStore(redis_url, prefix=f"test_shared_b_{uuid.uuid4().hex[:8]}", flush=True)
    try:
        t0 = datetime.now(timezone.utc) - timedelta(days=1)
        payee = f"shared-payee-{uuid.uuid4().hex[:8]}@ybl"
        for i in range(3):
            setup_event = make_event(txn_id=f"shared-setup-{i}", payee_vpa=payee, timestamp=t0 + timedelta(minutes=i))
            store_a.record(setup_event)
            store_b.record(setup_event)

        target_event = make_event(
            txn_id=f"shared-target-{uuid.uuid4().hex[:8]}", payee_vpa=payee, timestamp=t0 + timedelta(hours=1))

        api_result = scoring.score_event(target_event, store_a)

        fields = encode_event(target_event)
        worker_result = _score_and_log("0-1", fields, store_b, decision_log, source="worker")

        assert api_result.risk_score == worker_result.risk_score
        assert api_result.raw_score == worker_result.raw_score
        assert api_result.action == worker_result.action
        # NaN-aware comparison (feature_lib.tests.conftest): a plain `==` on
        # dicts containing float('nan') always fails, since NaN != NaN by
        # IEEE 754 definition, even when "both are NaN" is the correct,
        # matching outcome (e.g. amount_zscore_vs_payer_30d for a payee with
        # insufficient history).
        assert_vectors_equal(api_result.feature_snapshot, worker_result.feature_snapshot)
    finally:
        store_a.close()
        store_b.close()
