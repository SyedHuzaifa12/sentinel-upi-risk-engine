from worker.consumer import ensure_group, reclaim_pending
from worker.streams import DEAD_LETTER_STREAM, GROUP_NAME, MAX_RETRIES, STREAM_NAME


def test_forced_failure_is_not_acked_then_dead_lettered_after_max_retries(redis_client, store, decision_log):
    ensure_group(redis_client)

    # A malformed message: decode_event() will raise json.JSONDecodeError on
    # it, forcing a real scoring failure (not a mocked one).
    redis_client.xadd(STREAM_NAME, {"event": "not-valid-json"})

    # Deliver it once so it enters the pending entries list (PEL), without acking --
    # XAUTOCLAIM only reclaims entries already in some consumer's PEL.
    redis_client.xreadgroup(GROUP_NAME, "initial-consumer", {STREAM_NAME: ">"}, count=1, block=2000)

    pending_before = redis_client.xpending(STREAM_NAME, GROUP_NAME)
    assert pending_before["pending"] == 1  # confirmed NOT acked after the failed first delivery

    action_counts = {}
    for _attempt in range(MAX_RETRIES):
        # min_idle_time=0: reclaim immediately regardless of real idle time,
        # so the test doesn't have to wait out CLAIM_IDLE_MS between retries.
        reclaim_pending(redis_client, "reclaimer", store, decision_log, action_counts, min_idle_time=0)

    dead_letters = redis_client.xrange(DEAD_LETTER_STREAM, "-", "+")
    assert len(dead_letters) == 1

    pending_after = redis_client.xpending(STREAM_NAME, GROUP_NAME)
    assert pending_after["pending"] == 0  # dead-lettering also acks the original -- stops it being reclaimed forever
