"""The monitoring dashboard. Every data-bearing query here is raw SQL loaded
from sql/*.sql and executed via django.db.connection.cursor() -- Django's own
raw-SQL execution path, not the ORM's queryset/model layer. This is
deliberate: these are the SQL artifacts for this project, kept as plain,
readable, commented .sql files rather than buried in ORM method chains.
"""
import json
from pathlib import Path

from django.contrib.auth.decorators import login_required
from django.db import connection
from django.shortcuts import render

from ml.src.utils.paths import ARTIFACTS_DIR, FEATURE_REGISTRY_PATH, THRESHOLDS_PATH

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
    })
