import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest  # noqa: E402

from decisionlog.writer import DecisionLog  # noqa: E402


def _dsn():
    return __import__("os").environ.get("TEST_DATABASE_URL")


@pytest.fixture()
def decision_log():
    dsn = _dsn()
    if not dsn:
        pytest.skip("TEST_DATABASE_URL not set -- no Postgres reachable in this environment")
    log = DecisionLog(dsn)
    log._conn.execute("TRUNCATE decisions")
    try:
        yield log
    finally:
        log.close()


def sample_decision(txn_id="txn-1", **overrides):
    fields = dict(
        txn_id=txn_id,
        event={"payer_vpa": "alice@okaxis", "payee_vpa": "shop@ybl", "amount": 500.0},
        feature_snapshot={"amount_log": 6.2, "hour_of_day": 10},
        risk_score=0.02,
        raw_score=0.015,
        is_cold=True,
        action="ALLOW",
        risk_tier="LOW",
        reason_codes=[],
        model_version="cold-test",
        thresholds_version="testver1",
        latency_ms=5.0,
        scored_at=datetime.now(timezone.utc),
        source="api",
    )
    fields.update(overrides)
    return fields
