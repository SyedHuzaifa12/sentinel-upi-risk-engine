from worker.consumer import run
from worker.streams import STREAM_NAME, encode_event

from .conftest import make_event


def test_duplicate_delivery_of_the_same_event_creates_only_one_row(redis_client, store, decision_log):
    """Simulates an at-least-once redelivery: the exact same event (same
    txn_id) XADDed twice, both delivered and scored -- decisions.txn_id's
    UNIQUE constraint + ON CONFLICT DO NOTHING must keep this at one row."""
    event = make_event(txn_id="idempotent-1")
    redis_client.xadd(STREAM_NAME, encode_event(event))
    redis_client.xadd(STREAM_NAME, encode_event(event))

    result = run(redis_client, store, decision_log, "test-consumer-idem", max_messages=2, poll_block_ms=2000)

    assert result["processed"] == 2  # the worker DID process both deliveries...
    assert decision_log.count_by_txn_id("idempotent-1") == 1  # ...but only one row exists
