"""The backtest harness's core correctness property: re-scoring a decision
with its OWN model_version and the SAME thresholds.json that produced it
must reproduce the exact same action/risk_score, deterministically. This is
what proves the harness is sound (re-derives what already happened) rather
than merely "runs without crashing" -- swapping model/thresholds is only
trustworthy if the no-op case is provably a no-op.

Requires a live Postgres reachable via TEST_DATABASE_URL (same convention as
decisionlog/tests/conftest.py) and the currently on-disk cold/warm models
(ml/artifacts/models/*.pkl, matching ml/artifacts/models/feature_registry.json)
-- both already required for `make test` locally and in CI.

Import-order note (same as pipelines/backtest.py itself): `import lightgbm`
must be the literal first import, before anything that transitively imports
pandas.
"""
import lightgbm  # noqa: F401,E402,I001 -- MUST be imported before pandas, see module docstring
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest  # noqa: E402

from decisionlog.writer import DecisionLog  # noqa: E402
from feature_lib.store.in_memory import InMemoryHistoryStore  # noqa: E402
from ml.src.data.load_synthetic import load_synthetic_events  # noqa: E402
from ml.src.utils.paths import THRESHOLDS_PATH  # noqa: E402
from pipelines.backtest import run_backtest  # noqa: E402
from service import scoring  # noqa: E402


def _dsn():
    return os.environ.get("TEST_DATABASE_URL")


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


def test_backtest_reproduces_original_decision_exactly(decision_log, monkeypatch):
    monkeypatch.setattr("pipelines.backtest.DSN", _dsn())

    if not scoring.is_loaded():
        scoring.load_models()

    events = load_synthetic_events()
    store = InMemoryHistoryStore()

    # Score a short run-up of real events through the SAME store so the last
    # one is a realistic warm/cold case (mirrors how ml/src/train.py builds
    # its feature dataframe -- one store, one chronological pass).
    result = None
    for event in events[:20]:
        result = scoring.score_event(event, store)

    now = datetime.now(timezone.utc)
    decision_log.record(
        txn_id=result.txn_id,
        event={"amount": events[19].amount, "label_is_fraud": bool(events[19].label_is_fraud)},
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
        scored_at=now,
        source="api",
    )

    backtest_result = run_backtest(
        date_from=(now - timedelta(minutes=1)).isoformat(),
        date_to=(now + timedelta(minutes=1)).isoformat(),
        model_version=result.model_version,
        thresholds_path=THRESHOLDS_PATH,
    )

    a, b = backtest_result["scenario_a_original"], backtest_result["scenario_b_rescored"]
    assert a["action_mix"] == b["action_mix"]
    assert backtest_result["net_benefit_delta"] == pytest.approx(0.0, abs=1e-6)
