"""Consumes "txn.events" (consumer group "scorers"), scores each event
through service.scoring.score_event -- the SAME path the FastAPI service uses,
imported directly, never forked -- writes to the decision log, and XACKs.

    python -m worker.consumer

At-least-once handling: a message that raises during scoring is NOT XACKed.
It stays in the consumer group's pending entries list (PEL) until
`run()`'s reclaim pass (XAUTOCLAIM) picks it up again after CLAIM_IDLE_MS.
Each reclaim attempt increments a per-message retry counter in Redis; past
MAX_RETRIES the message is copied to the dead-letter stream and acked (so it
stops being reclaimed forever).

Idempotency does not need special handling here beyond what
decisionlog.DecisionLog.record() already does (ON CONFLICT (txn_id) DO
NOTHING) -- a redelivered message that scores successfully twice just writes
the same row once.

Graceful shutdown: SIGTERM/SIGINT set a flag checked BETWEEN batches, never
mid-batch -- whatever messages were already read in the current XREADGROUP
call are finished before the process exits.
"""
import argparse
import json
import logging
import os
import signal
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import redis  # noqa: E402

from decisionlog import DecisionLog  # noqa: E402
from feature_lib.store.redis_store import RedisHistoryStore  # noqa: E402
from service import scoring  # noqa: E402

from .streams import (  # noqa: E402
    CLAIM_IDLE_MS,
    DEAD_LETTER_STREAM,
    GROUP_NAME,
    MAX_RETRIES,
    RETRY_COUNT_KEY,
    STREAM_NAME,
    decode_event,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("worker.consumer")


class ShutdownFlag:
    def __init__(self):
        self.should_stop = False
        signal.signal(signal.SIGTERM, self._handle)
        signal.signal(signal.SIGINT, self._handle)

    def _handle(self, signum, _frame):
        logger.info("Received signal %s -- finishing in-flight work, then exiting.", signum)
        self.should_stop = True


def ensure_group(client: redis.Redis):
    try:
        client.xgroup_create(STREAM_NAME, GROUP_NAME, id="0", mkstream=True)
    except redis.ResponseError as exc:
        if "BUSYGROUP" not in str(exc):
            raise


def _score_and_log(msg_id, fields, store, decision_log, source="worker"):
    event = decode_event(fields)
    result = scoring.score_event(event, store)
    decision_log.record(
        txn_id=result.txn_id,
        event=json.loads(fields["event"]),
        feature_snapshot=result.feature_snapshot,
        risk_score=result.risk_score,
        raw_score=result.raw_score,
        is_cold=result.is_cold,
        action=result.action,
        risk_tier=result.risk_tier,
        reason_codes=result.reason_codes,
        model_version=result.model_version,
        thresholds_version=result.thresholds_version,
        latency_ms=result.latency_ms,
        scored_at=datetime.now(timezone.utc),
        source=source,
    )
    return result


def reclaim_pending(client: redis.Redis, consumer_name: str, store, decision_log, action_counts: dict,
                     min_idle_time: int = CLAIM_IDLE_MS) -> int:
    """XAUTOCLAIMs entries idle longer than `min_idle_time` ms. Retries each
    once; past MAX_RETRIES, dead-letters and acks. Returns count reclaimed.
    `min_idle_time` is a parameter (not just the module constant) so tests
    can reclaim immediately instead of waiting 30s."""
    reclaimed = 0
    cursor = "0-0"
    while True:
        cursor, claimed, _deleted = client.xautoclaim(
            STREAM_NAME, GROUP_NAME, consumer_name, min_idle_time=min_idle_time, start_id=cursor, count=50,
        )
        if not claimed:
            break
        for msg_id, fields in claimed:
            reclaimed += 1
            try:
                result = _score_and_log(msg_id, fields, store, decision_log)
                action_counts[result.action] = action_counts.get(result.action, 0) + 1
                client.xack(STREAM_NAME, GROUP_NAME, msg_id)
                client.hdel(RETRY_COUNT_KEY, msg_id)
            except Exception:
                retry_count = client.hincrby(RETRY_COUNT_KEY, msg_id, 1)
                logger.exception("Retry %d/%d failed for message %s", retry_count, MAX_RETRIES, msg_id)
                if retry_count >= MAX_RETRIES:
                    client.xadd(DEAD_LETTER_STREAM, dict(fields, _original_id=msg_id, _retries=str(retry_count)))
                    client.xack(STREAM_NAME, GROUP_NAME, msg_id)
                    client.hdel(RETRY_COUNT_KEY, msg_id)
                    logger.error("Message %s dead-lettered after %d retries", msg_id, retry_count)
        if cursor == "0-0":
            break
    return reclaimed


def run(client: redis.Redis, store, decision_log, consumer_name: str,
        max_messages=None, poll_block_ms=1000, report_interval_s=5.0, shutdown: ShutdownFlag = None,
        claim_idle_ms: int = CLAIM_IDLE_MS):
    """The main consume loop. `max_messages` bounds it for tests (None = run
    forever, checked via `shutdown`). `claim_idle_ms` is likewise overridable
    for tests that need to reclaim pending entries immediately."""
    ensure_group(client)
    action_counts = {}
    processed = 0
    iterations = 0
    t_start = time.perf_counter()
    last_report = t_start

    # max_iterations is a safety valve, not a normal termination path: a
    # message that fails every retry (and is dead-lettered by reclaim_pending,
    # which does NOT increment `processed`) would otherwise spin should_continue()
    # forever whenever the caller only bounds on max_messages -- this is what
    # caught a real bug (decisionlog NaN-vs-JSON) as a hang instead of a clean
    # assertion failure. Production callers (main()) don't pass max_messages
    # and rely on the SIGTERM `shutdown` flag instead, so this doesn't change
    # production behavior.
    max_iterations = None if max_messages is None else max(50, max_messages * 10)

    def should_continue():
        if max_messages is not None and processed >= max_messages:
            return False
        if max_iterations is not None and iterations >= max_iterations:
            logger.error("Stopping after %d iterations without reaching max_messages=%d "
                         "(processed=%d) -- a message is likely failing every attempt.",
                         iterations, max_messages, processed)
            return False
        if shutdown is not None and shutdown.should_stop:
            return False
        return True

    while should_continue():
        iterations += 1
        reclaim_pending(client, consumer_name, store, decision_log, action_counts, min_idle_time=claim_idle_ms)

        response = client.xreadgroup(GROUP_NAME, consumer_name, {STREAM_NAME: ">"}, count=10, block=poll_block_ms)
        if not response:
            continue

        for _stream_name, messages in response:
            for msg_id, fields in messages:
                try:
                    result = _score_and_log(msg_id, fields, store, decision_log)
                    action_counts[result.action] = action_counts.get(result.action, 0) + 1
                    client.xack(STREAM_NAME, GROUP_NAME, msg_id)
                    processed += 1
                except Exception:
                    logger.exception("Failed to score message %s -- leaving unacked for retry.", msg_id)
                if max_messages is not None and processed >= max_messages:
                    break

        now = time.perf_counter()
        if now - last_report > report_interval_s:
            elapsed = now - t_start
            logger.info("Throughput: %.1f events/sec | processed=%d | actions=%s",
                        processed / elapsed if elapsed else 0.0, processed, action_counts)
            last_report = now

    elapsed = time.perf_counter() - t_start
    logger.info("Consumer stopping. Processed=%d in %.1fs (%.1f events/sec) | actions=%s",
                processed, elapsed, processed / elapsed if elapsed else 0.0, action_counts)
    return {"processed": processed, "actions": action_counts, "elapsed_s": elapsed}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score events from the txn.events Redis Stream.")
    parser.add_argument("--redis-url", default=os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"))
    parser.add_argument("--consumer-name", default=None)
    args = parser.parse_args(argv)

    if not args.database_url:
        raise SystemExit("DATABASE_URL must be set (decisions are always logged to Postgres).")

    consumer_name = args.consumer_name or f"{socket.gethostname()}-{os.getpid()}"

    logger.info("Loading models (feature-registry guard runs first)...")
    scoring.load_models()

    client = redis.Redis.from_url(args.redis_url, decode_responses=True)
    client.ping()

    store = RedisHistoryStore(args.redis_url)
    decision_log = DecisionLog(args.database_url)
    shutdown = ShutdownFlag()

    logger.info("Worker %r consuming stream=%r group=%r", consumer_name, STREAM_NAME, GROUP_NAME)
    run(client, store, decision_log, consumer_name, shutdown=shutdown)


if __name__ == "__main__":
    main()
