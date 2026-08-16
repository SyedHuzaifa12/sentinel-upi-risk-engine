"""Replays data/synthetic/events.jsonl into the "txn.events" Redis Stream.

    python -m worker.producer --speed 0      # as fast as possible
    python -m worker.producer --speed 1      # real-time (sleeps the actual
                                              #   gap between event timestamps)
    python -m worker.producer --speed 10     # 10x real-time
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import redis  # noqa: E402

from ml.src.data.load_synthetic import load_synthetic_events  # noqa: E402

from .streams import LAG_WARNING_THRESHOLD, STREAM_MAXLEN, STREAM_NAME, encode_event  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("worker.producer")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Replay synthetic events into Redis Streams.")
    parser.add_argument("--speed", type=float, default=1.0,
                         help="0 = as fast as possible; 1 = real-time; N = N times real-time.")
    parser.add_argument("--redis-url", default=os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    parser.add_argument("--limit", type=int, default=None,
                         help="Replay only the first N events (default: the whole file) -- "
                              "useful for bounded demo/backfill runs.")
    args = parser.parse_args(argv)

    client = redis.Redis.from_url(args.redis_url, decode_responses=True)
    client.ping()

    logger.info("Loading synthetic events...")
    events = load_synthetic_events()
    if args.limit is not None:
        events = events[:args.limit]
    logger.info("Replaying %d events into stream %r at speed=%s", len(events), STREAM_NAME, args.speed)

    t_start = time.perf_counter()
    prev_ts = None
    for i, event in enumerate(events):
        if args.speed > 0 and prev_ts is not None:
            gap_seconds = (event.timestamp - prev_ts).total_seconds()
            if gap_seconds > 0:
                time.sleep(gap_seconds / args.speed)
        prev_ts = event.timestamp

        client.xadd(STREAM_NAME, encode_event(event), maxlen=STREAM_MAXLEN, approximate=True)

        if (i + 1) % 5000 == 0:
            xlen = client.xlen(STREAM_NAME)
            elapsed = time.perf_counter() - t_start
            logger.info("Produced %d/%d events (%.0f events/sec, stream length=%d)",
                        i + 1, len(events), (i + 1) / elapsed if elapsed else 0, xlen)
            if xlen > LAG_WARNING_THRESHOLD:
                logger.warning(
                    "Stream length %d exceeds %d -- consumer(s) may be falling behind.",
                    xlen, LAG_WARNING_THRESHOLD,
                )

    elapsed = time.perf_counter() - t_start
    logger.info("Done. Produced %d events in %.1fs (%.0f events/sec).",
                len(events), elapsed, len(events) / elapsed if elapsed else 0)


if __name__ == "__main__":
    main()
