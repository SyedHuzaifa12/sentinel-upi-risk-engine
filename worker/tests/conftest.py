import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest  # noqa: E402
import redis  # noqa: E402

from decisionlog import DecisionLog  # noqa: E402
from feature_lib.event import UPIEvent  # noqa: E402
from feature_lib.store.redis_store import RedisHistoryStore  # noqa: E402
from service import scoring  # noqa: E402
from worker.streams import DEAD_LETTER_STREAM, RETRY_COUNT_KEY, STREAM_NAME  # noqa: E402


def _test_redis_url():
    return os.environ.get("TEST_REDIS_URL")


def _test_database_url():
    return os.environ.get("TEST_DATABASE_URL")


@pytest.fixture()
def redis_client():
    url = _test_redis_url()
    if not url:
        pytest.skip("TEST_REDIS_URL not set -- no Redis reachable in this environment")
    client = redis.Redis.from_url(url, decode_responses=True)
    client.ping()
    for key in (STREAM_NAME, DEAD_LETTER_STREAM, RETRY_COUNT_KEY):
        client.delete(key)
    try:
        yield client
    finally:
        for key in (STREAM_NAME, DEAD_LETTER_STREAM, RETRY_COUNT_KEY):
            client.delete(key)
        client.close()


@pytest.fixture()
def store(redis_client):
    url = _test_redis_url()
    s = RedisHistoryStore(url, prefix=f"test_worker_{uuid.uuid4().hex[:8]}", flush=True)
    try:
        yield s
    finally:
        s.close()


@pytest.fixture()
def decision_log():
    dsn = _test_database_url()
    if not dsn:
        pytest.skip("TEST_DATABASE_URL not set -- no Postgres reachable in this environment")
    log = DecisionLog(dsn)
    log._conn.execute("TRUNCATE decisions")
    try:
        yield log
    finally:
        log.close()


@pytest.fixture(autouse=True, scope="session")
def _load_models_once():
    if not scoring.is_loaded():
        scoring.load_models()


def make_event(txn_id=None, payee_vpa="worker-test-payee@ybl", **overrides):
    fields = dict(
        timestamp=datetime.now(timezone.utc),
        payer_vpa="worker-test-payer@okaxis",
        amount=250.0,
        txn_type="P2P",
        initiation_mode="INTENT",
        device_id="dev-worker-test",
        payer_bank="AXIS",
        payee_bank="YBL",
        payer_account_age_days=365,
        label_is_fraud=False,
        label_typology=None,
    )
    fields.update(overrides)
    return UPIEvent(txn_id=txn_id or f"worker-test-{uuid.uuid4().hex[:12]}", payee_vpa=payee_vpa, **fields)
