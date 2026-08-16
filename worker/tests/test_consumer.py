from worker.consumer import run
from worker.streams import STREAM_NAME, encode_event

from .conftest import make_event


def test_one_message_produces_exactly_one_decision_row(redis_client, store, decision_log):
    event = make_event(txn_id="consumer-basic-1")
    redis_client.xadd(STREAM_NAME, encode_event(event))

    result = run(redis_client, store, decision_log, "test-consumer-1", max_messages=1, poll_block_ms=2000)

    assert result["processed"] == 1
    assert decision_log.count_by_txn_id("consumer-basic-1") == 1
    row = decision_log.fetch_by_txn_id("consumer-basic-1")
    assert row["source"] == "worker"
    assert row["action"] in ("ALLOW", "WARN", "REVIEW", "BLOCK")
