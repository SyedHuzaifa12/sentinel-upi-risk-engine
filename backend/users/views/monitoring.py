"""The monitoring dashboard. Every data-bearing query here is raw SQL loaded
from sql/*.sql and executed via django.db.connection.cursor() -- Django's own
raw-SQL execution path, not the ORM's queryset/model layer. This is
deliberate: these are the SQL artifacts for this project, kept as plain,
readable, commented .sql files rather than buried in ORM method chains.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db import connection
from django.shortcuts import redirect, render
from django.urls import reverse
from django.views.decorators.http import require_POST

from ml.src.utils.paths import ARTIFACTS_DIR, FEATURE_REGISTRY_PATH, THRESHOLDS_PATH

REPLAY_N_EVENTS = 100

REPO_ROOT = Path(__file__).resolve().parents[3]
SQL_DIR = REPO_ROOT / "sql"


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


@login_required
@require_POST
def replay_events(request):
    """Scores REPLAY_N_EVENTS fresh synthetic events in-process (Render
    deploy target only -- see _replay_state()) against the shared store and
    decision log already loaded by render_app.py's combined lifespan. A
    small payer/payee pool (not a fresh one per click) so payees recur
    WITHIN a single 100-event batch, giving realistic warm-model routing
    without needing to persist a separate identity pool across clicks --
    unlike training/backtesting, a live demo button intentionally varies its
    output per click (seeded from the current time), so this is not held to
    the project's usual "always a fixed seed" reproducibility rule."""
    state = _replay_state()
    if state is None:
        messages.error(
            request,
            "Live replay isn't available in this deployment -- it needs the combined "
            "Render process (render_app.py), not the docker-compose ui/api split.",
        )
        return redirect(reverse("monitoring_dashboard"))

    from ml.src.generator.generator import generate
    from service import scoring

    t_start = datetime.now(timezone.utc)
    seed = int(t_start.timestamp())  # varies per click, deliberately -- see docstring above
    events, _summary = generate(days=1, n_payers=15, n_payees=8, seed=seed, fraud_rate=0.03)
    events = events[:REPLAY_N_EVENTS]

    action_counts = {}
    n_fraud_caught = 0
    for event in events:
        result = scoring.score_event(event, state.store)
        action_counts[result.action] = action_counts.get(result.action, 0) + 1
        if event.label_is_fraud and result.action != "ALLOW":
            n_fraud_caught += 1
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

    elapsed_s = (datetime.now(timezone.utc) - t_start).total_seconds()
    messages.success(
        request,
        f"Replayed {len(events)} events in {elapsed_s:.1f}s -- {action_counts}, "
        f"{n_fraud_caught} fraud caught. (Free-tier 0.1 CPU: this is expected to be "
        f"much slower than the 27ms p99 measured locally with dedicated hardware.)",
    )
    return redirect(reverse("monitoring_dashboard"))
