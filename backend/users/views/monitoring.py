"""The monitoring dashboard. Every data-bearing query here is raw SQL loaded
from sql/*.sql and executed via django.db.connection.cursor() -- Django's own
raw-SQL execution path, not the ORM's queryset/model layer. This is
deliberate: these are the SQL artifacts for this project, kept as plain,
readable, commented .sql files rather than buried in ORM method chains.
"""
import json
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path

from django.contrib.auth.decorators import login_required
from django.db import connection
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST

from ml.src.utils.paths import ARTIFACTS_DIR, FEATURE_REGISTRY_PATH, THRESHOLDS_PATH

REPLAY_N_EVENTS = 25

REPO_ROOT = Path(__file__).resolve().parents[3]
SQL_DIR = REPO_ROOT / "sql"

# Module-level, process-wide job state (2026-08-19, replacing a synchronous
# view that returned a live 500). Safe ONLY because this deploy target runs
# ONE uvicorn process with no --workers flag (see render_entrypoint.sh) --
# a module-level dict would NOT be shared across multiple worker processes.
# Guarded by a lock for the start/check-if-running race; individual field
# writes from the background thread don't need it (CPython dict item
# assignment is already atomic under the GIL, and nothing here does a
# compound read-modify-write across two fields that must stay in sync).
_replay_lock = threading.Lock()
_replay_job = {
    "running": False, "scored": 0, "total": 0, "action_counts": {}, "n_fraud_caught": 0,
    "error": None, "done": False, "started_at": None, "elapsed_s": None,
}


def _run_sql_file(name: str) -> list:
    sql_text = (SQL_DIR / name).read_text()
    with connection.cursor() as cursor:
        cursor.execute(sql_text)
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _current_versions():
    try:
        with open(FEATURE_REGISTRY_PATH) as f:
            registry = json.load(f)
        cold_version = registry["cold_model"]["model_version"]
        warm_version = registry["warm_model"]["model_version"]
    except FileNotFoundError:
        cold_version = warm_version = "unknown"

    try:
        import hashlib
        with open(THRESHOLDS_PATH, "rb") as f:
            thresholds_version = hashlib.sha256(f.read()).hexdigest()[:8]
    except FileNotFoundError:
        thresholds_version = "unknown"

    return cold_version, warm_version, thresholds_version


def _latest_drift_report():
    """PSI's training-distribution reference lives in the offline synthetic
    dataset, not Postgres -- this reads the most recently written
    drift_<ts>.json rather than computing PSI as a live SQL query."""
    drift_dir = ARTIFACTS_DIR / "drift"
    if not drift_dir.exists():
        return None
    files = sorted(drift_dir.glob("drift_*.json"))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


@login_required
def monitoring_dashboard(request):
    alert_rate = _run_sql_file("alert_rate_over_time.sql")
    action_mix = _run_sql_file("action_mix.sql")
    score_histogram = _run_sql_file("score_histogram.sql")
    latency = _run_sql_file("latency_percentiles.sql")
    queue = _run_sql_file("review_queue_depth_and_age.sql")
    precision = _run_sql_file("precision_on_reviewed_cases.sql")

    cold_version, warm_version, thresholds_version = _current_versions()
    drift_report = _latest_drift_report()

    top_drifted = []
    cold_start_caveat = None
    if drift_report:
        feature_drift = drift_report.get("feature_drift", {})
        top_drifted = sorted(feature_drift.get("psi_by_feature", {}).items(), key=lambda kv: kv[1], reverse=True)[:10]
        cold_start_caveat = feature_drift.get("cold_start_caveat")

    return render(request, "app/monitoring.html", {
        "alert_rate": alert_rate,
        "action_mix": action_mix,
        "score_histogram": score_histogram,
        "latency": latency,
        "queue": queue[0] if queue else None,
        "precision": precision[0] if precision else None,
        "cold_version": cold_version,
        "warm_version": warm_version,
        "thresholds_version": thresholds_version,
        "top_drifted": top_drifted,
        "cold_start_caveat": cold_start_caveat,
        "drift_report_generated_at": drift_report.get("generated_at_utc") if drift_report else None,
        "replay_available": _replay_state() is not None,
        "replay_n_events": REPLAY_N_EVENTS,
    })


def _replay_state():
    """The FastAPI sub-app's `.state.store`/`.state.decision_log`, but ONLY
    when this Django process is actually running inside render_app.py's
    combined ASGI process (Render deploy target) -- under docker-compose's
    separate ui/api containers, `service.main.app`'s own lifespan never ran
    in THIS process, so `.state` has no `store` attribute at all. Returns
    None in that case rather than raising, so the button degrades to a
    clear disabled state instead of a 500."""
    from service.main import app as fastapi_app
    if not hasattr(fastapi_app.state, "store") or fastapi_app.state.store is None:
        return None
    return fastapi_app.state


def _run_replay_job(state):
    """Runs entirely in a background thread -- see start_replay() below.
    Wrapped in try/except/finally as a whole: ANY failure (a scoring error,
    a dropped Neon connection mid-loop, an unexpected exception this wasn't
    specifically written to anticipate) must land in `_replay_job['error']`
    for the status endpoint to surface, never propagate and crash a
    request -- there IS no request by the time this runs, it's already
    been handed off, but a silent thread death with no error recorded would
    be just as unhelpful as the original bare 500 this replaces."""
    from ml.src.generator.generator import generate
    from service import scoring

    try:
        seed = int(datetime.now(timezone.utc).timestamp())  # varies per click, deliberately -- see start_replay()
        # days=3, not 1 (2026-08-19 bug, found via a live 500): the
        # generator's churn-window stratification does
        # `window_end = min(days - 1, ...)`, which is 0 when days=1 --
        # window_end < window_start guarantees `ValueError: low >= high`
        # in numpy's rng.integers on EVERY call, deterministically,
        # regardless of seed. days=3 gives the window real margin.
        events, _summary = generate(days=3, n_payers=15, n_payees=8, seed=seed, fraud_rate=0.03)
        events = events[:REPLAY_N_EVENTS]
        _replay_job["total"] = len(events)

        for event in events:
            result = scoring.score_event(event, state.store)
            counts = _replay_job["action_counts"]
            counts[result.action] = counts.get(result.action, 0) + 1
            if event.label_is_fraud and result.action != "ALLOW":
                _replay_job["n_fraud_caught"] += 1
            if state.decision_log is not None:
                state.decision_log.record(
                    txn_id=result.txn_id,
                    event={
                        "payer_vpa": event.payer_vpa, "payee_vpa": event.payee_vpa, "amount": event.amount,
                        "label_is_fraud": event.label_is_fraud, "label_typology": event.label_typology,
                    },
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
                    scored_at=datetime.now(timezone.utc),
                    source="replay",
                )
            _replay_job["scored"] += 1
    except Exception as exc:  # noqa: BLE001 -- deliberately broad, see docstring
        _replay_job["error"] = f"{type(exc).__name__}: {exc}"
        print(f"monitoring.replay_events: background job failed: {exc}\n{traceback.format_exc()}")
    finally:
        started_at = _replay_job["started_at"]
        _replay_job["elapsed_s"] = (
            (datetime.now(timezone.utc) - started_at).total_seconds() if started_at else None
        )
        _replay_job["done"] = True
        _replay_job["running"] = False


@login_required
@require_POST
def start_replay(request):
    """Starts the replay job in a background thread and returns immediately
    -- the page never blocks on ~1s/event x N on a 0.1 CPU box, and never
    holds the HTTP request open long enough to hit a platform timeout.
    `monitoring.html`'s small polling script drives the visible progress
    via replay_status() below and reloads once `done` is true."""
    state = _replay_state()
    if state is None:
        return JsonResponse({
            "error": "Live replay isn't available in this deployment -- it needs the combined "
                     "Render process (render_app.py), not the docker-compose ui/api split.",
        }, status=409)

    with _replay_lock:
        if _replay_job["running"]:
            return JsonResponse({"error": "A replay is already running."}, status=409)
        _replay_job.update(
            running=True, scored=0, total=0, action_counts={}, n_fraud_caught=0,
            error=None, done=False, started_at=datetime.now(timezone.utc), elapsed_s=None,
        )
        thread = threading.Thread(target=_run_replay_job, args=(state,), daemon=True)
        thread.start()

    return JsonResponse({"started": True})


@login_required
def replay_status(request):
    """Polled by monitoring.html every ~1.5s while a replay is running."""
    job = dict(_replay_job)
    job["started_at"] = job["started_at"].isoformat() if job["started_at"] else None
    return JsonResponse(job)
