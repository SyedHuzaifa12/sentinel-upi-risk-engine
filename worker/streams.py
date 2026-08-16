"""Shared Redis Streams constants + event (de)serialization, used by
producer.py, consumer.py, and their tests -- one encoding, not reimplemented
per module.
"""
import json

from feature_lib.event import UPIEvent

STREAM_NAME = "txn.events"
DEAD_LETTER_STREAM = "txn.events.dead"
GROUP_NAME = "scorers"

# Approximate trimming (~): bounds stream memory against Redis's
# maxmemory=256mb/noeviction cap so a fast producer outrunning the worker
# doesn't start erroring on XADD instead of just trimming old entries.
STREAM_MAXLEN = 100_000
# Logged as a warning (not enforced) when XLEN exceeds this -- signals
# consumer lag rather than failing silently.
LAG_WARNING_THRESHOLD = 50_000

RETRY_COUNT_KEY = "txn.events:retries"
MAX_RETRIES = 3
CLAIM_IDLE_MS = 30_000  # how long a pending entry sits unacked before XAUTOCLAIM reclaims it


def encode_event(event: UPIEvent) -> dict:
    return {"event": event.model_dump_json()}


def decode_event(fields: dict) -> UPIEvent:
    return UPIEvent(**json.loads(fields["event"]))
