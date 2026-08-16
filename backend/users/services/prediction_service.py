"""Client for the Phase 4 FastAPI scoring service.

Django no longer loads a model in-process -- it POSTs a raw event to
`/v1/score` and renders whatever comes back. Timeout is short (2s) with one
manual retry, and any failure raises RiskAPIUnavailable rather than a bare
500 or a silent fallback to the (now-legacy, archived) RandomForest model.
"""
import httpx
from django.conf import settings

_RETRYABLE_EXCEPTIONS = (httpx.TimeoutException, httpx.ConnectError)


class RiskAPIUnavailable(Exception):
    """Raised when the scoring service can't be reached after one retry, or
    returns a server error. Callers must show a clear degraded-mode message,
    never a 500 and never a silent fallback to a different model."""


def predict_event(event: dict) -> dict:
    """POST a raw UPI event dict to /v1/score. Returns the parsed JSON
    response (txn_id, risk_score, action, risk_tier, reason_codes,
    model_version, thresholds_version, is_cold, latency_ms,
    features_computed) on success.

    Raises RiskAPIUnavailable on timeout/connection failure (after one
    retry) or a 5xx response. A 4xx response (e.g. a malformed event) is
    treated as a genuine client-side error and re-raised as
    RiskAPIUnavailable too -- the sandbox form's own input constraints
    should prevent this in practice, but there is no legacy model left to
    silently fall back to.
    """
    last_exc = None
    for _attempt in range(2):
        try:
            with httpx.Client(base_url=settings.RISK_API_URL, timeout=2.0) as client:
                response = client.post("/v1/score", json=event)
            response.raise_for_status()
            return response.json()
        except _RETRYABLE_EXCEPTIONS as exc:
            last_exc = exc
            continue
        except httpx.HTTPStatusError as exc:
            raise RiskAPIUnavailable(f"Risk scoring service returned {exc.response.status_code}") from exc

    raise RiskAPIUnavailable("Risk scoring service is unreachable") from last_exc
