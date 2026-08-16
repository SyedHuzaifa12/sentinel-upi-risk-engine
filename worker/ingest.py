"""Batch/CSV ingest -- the Phase 6 backtest entry point.

    python -m worker.ingest --file events.jsonl
    python -m worker.ingest --file events.csv

Reads a JSONL or CSV file, sorts by timestamp, scores every row through
service.scoring.score_event (the SAME scoring path the API and the Streams
worker use), and writes each decision with source='replay'. Uses a fresh
InMemoryHistoryStore built up in timestamp order -- this is a one-shot batch
process building its own local history, exactly like ml/src/train.py's own
`build_feature_dataframe` pattern, not a shared online store (there's no
reason for a one-off replay to touch the live Redis-backed history).
"""
import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decisionlog import DecisionLog  # noqa: E402
from feature_lib.event import UPIEvent  # noqa: E402
from feature_lib.store.in_memory import InMemoryHistoryStore  # noqa: E402
from ml.src.data.load_synthetic import load_synthetic_events  # noqa: E402
from service import scoring  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("worker.ingest")


def _load_csv(path) -> list:
    events = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            events.append(UPIEvent(
                txn_id=row["txn_id"],
                timestamp=row["timestamp"],
                payer_vpa=row["payer_vpa"],
                payee_vpa=row["payee_vpa"],
                amount=float(row["amount"]),
                txn_type=row["txn_type"],
                initiation_mode=row["initiation_mode"],
                device_id=row["device_id"],
                payer_bank=row["payer_bank"],
                payee_bank=row["payee_bank"],
                payer_account_age_days=int(row["payer_account_age_days"]),
                label_is_fraud=row.get("label_is_fraud", "").strip().lower() in ("true", "1", "y", "yes"),
                label_typology=row.get("label_typology") or None,
            ))
    return events


def load_events(path: Path) -> list:
    if path.suffix == ".csv":
        events = _load_csv(path)
    elif path.suffix in (".jsonl", ".json"):
        events = load_synthetic_events(path)
    else:
        raise ValueError(f"Unsupported file extension {path.suffix!r} -- expected .jsonl or .csv")
    return sorted(events, key=lambda e: e.timestamp)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Batch-score a JSONL/CSV file of raw UPI events.")
    parser.add_argument("--file", required=True, type=Path)
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"))
    args = parser.parse_args(argv)

    if not args.database_url:
        raise SystemExit("DATABASE_URL must be set (decisions are always logged to Postgres).")

    logger.info("Loading models (feature-registry guard runs first)...")
    scoring.load_models()

    events = load_events(args.file)
    logger.info("Loaded %d events from %s, scoring in timestamp order...", len(events), args.file)

    store = InMemoryHistoryStore()
    decision_log = DecisionLog(args.database_url)

    action_counts = {}
    t_start = time.perf_counter()
    for i, event in enumerate(events):
        result = scoring.score_event(event, store)
        action_counts[result.action] = action_counts.get(result.action, 0) + 1
        decision_log.record(
            txn_id=result.txn_id,
            event=json.loads(event.model_dump_json()),
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
            source="replay",
        )
        if (i + 1) % 5000 == 0:
            logger.info("Scored %d/%d events... actions=%s", i + 1, len(events), action_counts)

    elapsed = time.perf_counter() - t_start
    logger.info("Done. Scored %d events in %.1fs (%.0f events/sec) | actions=%s",
                len(events), elapsed, len(events) / elapsed if elapsed else 0, action_counts)


if __name__ == "__main__":
    main()
