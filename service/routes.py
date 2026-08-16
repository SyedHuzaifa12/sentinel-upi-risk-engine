import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request

from . import scoring
from .deps import get_store
from .schemas import HealthResponse, ModelInfoResponse, ScoreRequest, ScoreResponse, to_upi_event

router = APIRouter()


@router.post("/v1/score", response_model=ScoreResponse)
def score(body: ScoreRequest, request: Request, store=Depends(get_store)):
    event = to_upi_event(body)
    result = scoring.score_event(event, store)

    request.app.state.logger.info(
        "scored txn_id=%s action=%s is_cold=%s stage_latency_ms=%s total_ms=%.2f",
        result.txn_id, result.action, result.is_cold, result.stage_latency_ms, result.latency_ms,
    )

    return ScoreResponse(
        txn_id=result.txn_id,
        risk_score=result.risk_score,
        action=result.action,
        risk_tier=result.risk_tier,
        reason_codes=result.reason_codes,
        model_version=result.model_version,
        thresholds_version=result.thresholds_version,
        is_cold=result.is_cold,
        latency_ms=result.latency_ms,
        features_computed=result.features_computed,
    )


@router.get("/v1/health", response_model=HealthResponse)
def health(request: Request, store=Depends(get_store)):
    store_reachable = True
    try:
        store.payee_txn_count("__healthcheck__", datetime.now(timezone.utc))
    except Exception:
        store_reachable = False

    return HealthResponse(
        status="ok" if scoring.is_loaded() and store_reachable else "degraded",
        cold_model_loaded=scoring.is_loaded(),
        warm_model_loaded=scoring.is_loaded(),
        store_backend=request.app.state.store_backend,
        store_reachable=store_reachable,
        uptime_seconds=time.time() - request.app.state.started_at_epoch,
    )


@router.get("/v1/model-info", response_model=ModelInfoResponse)
def model_info(request: Request):
    return request.app.state.model_info
