"""Same repo-root sys.path convention as feature_lib/tests/conftest.py and
ml/tests -- makes `feature_lib`/`ml`/`service` importable as top-level
packages regardless of how pytest is invoked."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from service.main import app  # noqa: E402


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def make_score_request(**overrides):
    body = {
        "txn_id": "test-txn-1",
        "timestamp": "2026-03-01T10:00:00Z",
        "payer_vpa": "alice@okaxis",
        "payee_vpa": "newmerchant@ybl",
        "amount": 500.0,
        "txn_type": "P2P",
        "initiation_mode": "INTENT",
        "device_id": "dev-abc",
        "payer_bank": "AXIS",
        "payee_bank": "YBL",
        "payer_account_age_days": 800,
    }
    body.update(overrides)
    return body
