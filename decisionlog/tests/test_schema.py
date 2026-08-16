"""decisions is append-only: UPDATE/DELETE must both raise, and a duplicate
txn_id insert must be a silent no-op, not a second row."""
import psycopg
import pytest

from .conftest import sample_decision


def test_duplicate_txn_id_insert_is_a_no_op(decision_log):
    decision_log.record(**sample_decision("dup-1"))
    decision_log.record(**sample_decision("dup-1", risk_score=0.99))  # different payload, same txn_id

    assert decision_log.count_by_txn_id("dup-1") == 1
    row = decision_log.fetch_by_txn_id("dup-1")
    assert row["risk_score"] == 0.02  # the FIRST insert wins; ON CONFLICT DO NOTHING


def test_update_is_rejected(decision_log):
    decision_log.record(**sample_decision("no-update-1"))
    with pytest.raises(psycopg.errors.RaiseException, match="append-only"):
        decision_log._conn.execute("UPDATE decisions SET risk_score = 0.5 WHERE txn_id = %s", ("no-update-1",))


def test_delete_is_rejected(decision_log):
    decision_log.record(**sample_decision("no-delete-1"))
    with pytest.raises(psycopg.errors.RaiseException, match="append-only"):
        decision_log._conn.execute("DELETE FROM decisions WHERE txn_id = %s", ("no-delete-1",))


def test_nan_feature_values_are_stored_as_json_null_not_rejected(decision_log):
    """Regression test: a real payer/payee with insufficient history produces
    NaN for several feature_lib features (e.g. amount_zscore_vs_payer_30d) --
    not an edge case, the common cold-start case. json.dumps(float('nan'))
    emits the non-standard `NaN` token, which Postgres's JSONB correctly
    rejects. Found by worker/tests' live integration tests, where every
    message failed to log until this was fixed."""
    decision_log.record(**sample_decision(
        "nan-feature-1",
        feature_snapshot={"amount_log": 6.2, "amount_zscore_vs_payer_30d": float("nan")},
    ))

    row = decision_log.fetch_by_txn_id("nan-feature-1")
    assert row["feature_snapshot"]["amount_zscore_vs_payer_30d"] is None
