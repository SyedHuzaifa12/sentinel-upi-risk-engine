"""The one shared decision-log writer. Both service/routes.py (API) and
worker/consumer.py (and worker/ingest.py) import and call this -- never a
second implementation of "write a decision row."
"""
import json
import subprocess
from pathlib import Path

import psycopg

from .schema import ensure_schema


def _json_safe(value):
    """Recursively replaces NaN/Infinity floats with None.

    Real bug this fixes: many feature_lib features are legitimately NaN (any
    `_is_missing`-eligible feature, e.g. amount_zscore_vs_payer_30d, is NaN
    for a payer/payee with insufficient history -- the common case for a
    brand-new payee, not an edge case). Python's json.dumps happily emits the
    non-standard `NaN` token for float('nan'), but that is not valid JSON --
    Postgres's JSONB column correctly rejects it
    (psycopg.errors.InvalidTextRepresentation: "Token 'NaN' is invalid").
    Without this, decision logging would fail for most real cold-start
    transactions. Caught by worker/tests' live-Postgres integration tests,
    not by any unit test with a hand-picked "nice" feature vector.
    """
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):  # NaN never equals itself
            return None
        return value
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _git_sha() -> str:
    """feature_lib doesn't carry its own semantic version -- the repo's git
    SHA at scoring time is the honest, zero-maintenance stand-in (same
    pattern ml/src/train.py already uses for model_version)."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).resolve().parents[1],
        ).decode().strip()
    except Exception:
        return "unknown"


class DecisionLog:
    def __init__(self, dsn: str):
        self._conn = psycopg.connect(dsn, autocommit=True)
        ensure_schema(self._conn)
        self._feature_lib_version = _git_sha()

    def close(self):
        self._conn.close()

    def record(self, *, txn_id, event: dict, feature_snapshot: dict, risk_score: float,
               raw_score: float, is_cold: bool, action: str, risk_tier: str, reason_codes: list,
               model_version: str, thresholds_version: str, latency_ms: float, scored_at, source: str) -> None:
        """Insert one decision row. Idempotent on txn_id (ON CONFLICT DO
        NOTHING) -- a duplicate delivery (e.g. an at-least-once Streams
        redelivery) never creates a second row, mirroring
        PostgresHistoryStore.record()'s existing idempotency pattern."""
        self._conn.execute(
            """
            INSERT INTO decisions
                (txn_id, event, feature_snapshot, risk_score, raw_score, is_cold, action,
                 risk_tier, reason_codes, model_version, thresholds_version,
                 feature_lib_version, latency_ms, scored_at, source)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (txn_id) DO NOTHING
            """,
            (
                txn_id, json.dumps(_json_safe(event), default=str),
                json.dumps(_json_safe(feature_snapshot), default=str),
                risk_score, raw_score, is_cold, action, risk_tier, json.dumps(_json_safe(reason_codes)),
                model_version, thresholds_version, self._feature_lib_version, latency_ms,
                scored_at, source,
            ),
        )

    def fetch_by_txn_id(self, txn_id: str):
        row = self._conn.execute(
            "SELECT txn_id, event, feature_snapshot, risk_score, raw_score, is_cold, action, "
            "risk_tier, reason_codes, model_version, thresholds_version, feature_lib_version, "
            "latency_ms, scored_at, source FROM decisions WHERE txn_id = %s",
            (txn_id,),
        ).fetchone()
        if row is None:
            return None
        columns = ("txn_id", "event", "feature_snapshot", "risk_score", "raw_score", "is_cold",
                   "action", "risk_tier", "reason_codes", "model_version", "thresholds_version",
                   "feature_lib_version", "latency_ms", "scored_at", "source")
        return dict(zip(columns, row))

    def count_by_txn_id(self, txn_id: str) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM decisions WHERE txn_id = %s", (txn_id,)).fetchone()
        return row[0]
