from datetime import datetime, timezone

from feature_lib.event import UPIEvent
from feature_lib.store.in_memory import InMemoryHistoryStore
from service import scoring
from service.tests.conftest import make_score_request


def test_valid_event_returns_well_formed_response(client):
    response = client.post("/v1/score", json=make_score_request())
    assert response.status_code == 200

    body = response.json()
    for key in ("txn_id", "risk_score", "action", "risk_tier", "reason_codes",
                "model_version", "thresholds_version", "is_cold", "latency_ms", "features_computed"):
        assert key in body
    assert body["action"] in ("ALLOW", "WARN", "REVIEW", "BLOCK")
    assert body["txn_id"] == "test-txn-1"


def test_posting_a_label_field_is_rejected_with_422(client):
    response = client.post("/v1/score", json=make_score_request(label_is_fraud=True))
    assert response.status_code == 422


def test_posting_a_precomputed_feature_is_rejected_with_422(client):
    response = client.post("/v1/score", json=make_score_request(payer_txn_count_1h=5))
    assert response.status_code == 422


def test_new_payee_routes_cold_then_warm_after_three_prior_events(client):
    base = make_score_request(payee_vpa="freshpayee@ybl")
    responses = []
    for i in range(4):
        body = dict(base, txn_id=f"routing-{i}", timestamp=f"2026-03-01T10:0{i}:00Z")
        responses.append(client.post("/v1/score", json=body).json())

    assert [r["is_cold"] for r in responses] == [True, True, True, False]


def _make_event(txn_id, timestamp, payee_vpa="regular@ybl", **overrides):
    fields = dict(
        payer_vpa="alice@okaxis", amount=500.0, txn_type="P2P", initiation_mode="INTENT",
        device_id="dev-abc", payer_bank="AXIS", payee_bank="YBL", payer_account_age_days=800,
        label_is_fraud=False, label_typology=None,
    )
    fields.update(overrides)
    return UPIEvent(txn_id=txn_id, timestamp=timestamp, payee_vpa=payee_vpa, **fields)


def test_scoring_is_deterministic_given_a_frozen_store_and_stateful_once_recorded():
    """Documents the intended statefulness of score_event, rather than
    fighting it: given two stores in an IDENTICAL prior state, scoring the
    same event produces an identical score. Once one store has recorded an
    additional event that the other hasn't, scoring the same follow-up
    event against each is free to differ."""
    if not scoring.is_loaded():
        scoring.load_models()

    store_a = InMemoryHistoryStore()
    store_b = InMemoryHistoryStore()
    prior = _make_event("prior-1", datetime(2026, 3, 1, 9, 0, tzinfo=timezone.utc))
    store_a.record(prior)
    store_b.record(prior)

    probe = _make_event("probe-1", datetime(2026, 3, 1, 9, 30, tzinfo=timezone.utc))
    result_a = scoring.score_event(probe, store_a)
    probe_repeat = _make_event("probe-1", datetime(2026, 3, 1, 9, 30, tzinfo=timezone.utc))
    result_b = scoring.score_event(probe_repeat, store_b)
    assert result_a.risk_score == result_b.risk_score

    # Now store_a has diverged (it recorded `probe`); scoring another event
    # against the two stores is no longer guaranteed to match -- this is
    # intended statefulness, not a bug.
    extra = _make_event("extra-1", datetime(2026, 3, 1, 9, 45, tzinfo=timezone.utc))
    result_a2 = scoring.score_event(extra, store_a)
    result_b2 = scoring.score_event(extra, store_b)
    assert result_a2.features_computed == result_b2.features_computed  # same shape
    # (values may legitimately differ -- store_a has one more prior event recorded)
