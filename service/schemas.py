"""Pydantic request/response contracts for /v1/score.

Request: feature_lib.event.UPIEvent's own field definitions are reused
directly (not retyped by hand) via pydantic's create_model, minus the two
training-only label fields -- a real caller never has (or should send)
label_is_fraud/label_typology. extra="forbid" means posting a precomputed
feature name, a label field, or anything else not in the raw event schema
is a 422 from Pydantic's own validation, with no manual rejection logic
needed here.
"""
from typing import Optional

from pydantic import BaseModel, ConfigDict, create_model

from feature_lib.event import UPIEvent

_SCORE_REQUEST_FIELDS = {
    name: (field_info.annotation, field_info)
    for name, field_info in UPIEvent.model_fields.items()
    if name not in ("label_is_fraud", "label_typology")
}

ScoreRequest = create_model(
    "ScoreRequest",
    __config__=ConfigDict(extra="forbid"),
    **_SCORE_REQUEST_FIELDS,
)


def to_upi_event(request) -> UPIEvent:
    return UPIEvent(**request.model_dump(), label_is_fraud=False, label_typology=None)


class ScoreResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    txn_id: str
    risk_score: float
    action: str
    risk_tier: str
    reason_codes: list
    model_version: str
    thresholds_version: str
    is_cold: bool
    latency_ms: float
    features_computed: int


class HealthResponse(BaseModel):
    status: str
    cold_model_loaded: bool
    warm_model_loaded: bool
    store_backend: str
    store_reachable: bool
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    cold_model_version: str
    warm_model_version: str
    thresholds_version: str
    cold_feature_count: int
    warm_feature_count: int
    trained_at_utc: Optional[str] = None
