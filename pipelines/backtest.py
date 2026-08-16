"""The backtest harness -- re-scores a historical window of real decisions
using a candidate model/thresholds pair, WITHOUT recomputing features.

    python -m pipelines.backtest --from 2026-01-01 --to 2026-01-31 \
        --model warm-20260816-9a6747c --thresholds ml/artifacts/policy/thresholds.json
    make backtest

Every row in `decisions` already carries its full `feature_snapshot` (the
exact computed vector that produced the original score) -- that snapshot is
what lets this harness swap the model or the thresholds and see what WOULD
have happened, without needing to reconstruct historical HistoryStore state
(expensive, and not exactly reproducible even if you tried). This is the
entire reason the snapshot column exists.

Ground truth: `decisions.event->>'label_is_fraud'` is present for worker-
sourced rows (worker/producer.py replays full UPIEvents, labels included) but
absent for API-sourced rows (the live ScoreRequest schema deliberately
excludes labels a real caller shouldn't send). Rows without a label are
excluded from ground-truth-dependent metrics and counted separately -- never
silently treated as "not fraud".

Import-order note (load-bearing, do not reorder -- same issue documented in
ml/src/policy/reason_codes.py and service/scoring.py): if `pandas` is
imported before the bare `lightgbm` package anywhere in this process,
LightGBM's ctypes bridge crashes with a native access-violation the first
time its C API is touched. `import lightgbm` must happen before
`feature_lib.frame` (which imports pandas) -- `import mlflow.lightgbm` does
NOT guarantee this, since mlflow may import the bare package lazily.
"""
import lightgbm  # noqa: F401,E402,I001 -- MUST be imported before pandas, see module docstring
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import mlflow  # noqa: E402
import mlflow.artifacts  # noqa: E402
import mlflow.lightgbm  # noqa: E402
import numpy as np  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from decisionlog import DecisionLog  # noqa: E402
from feature_lib.frame import vector_to_frame  # noqa: E402
from feature_lib.registry import ALL_FEATURES, COLD_FEATURES  # noqa: E402
from ml.src.policy.cost_matrix import FN_COST, FP_COST_BLOCK, FP_COST_FRICTION, FP_COST_REVIEW, TP_BENEFIT  # noqa: E402
from ml.src.policy.decide import _action_for_tier, _risk_tier  # noqa: E402
from ml.src.training.lightgbm_pipeline import apply_calibrator  # noqa: E402
from ml.src.utils.paths import (  # noqa: E402
    ARTIFACTS_DIR,
    COLD_CALIBRATOR_PATH,
    COLD_MODEL_PATH,
    FEATURE_REGISTRY_PATH,
    WARM_CALIBRATOR_PATH,
    WARM_MODEL_PATH,
)
from pipelines.train import MLRUNS_DIR  # noqa: E402

BACKTEST_DIR = ARTIFACTS_DIR / "backtests"
# Same DATABASE_URL convention config/settings.py and docker-compose.yml use;
# falls back to the local-Docker default for ad hoc runs outside a container.
DSN = os.environ.get("DATABASE_URL", "postgresql://sentinel:sentinel@localhost:5432/sentinel")


def _load_model_and_calibrator(model_version: str, is_cold: bool):
    """Resolves `model_version` to an actual model+calibrator. Primary path:
    the MLflow registry (via the model_version TAG pipelines/train.py set --
    this is what the registry linkage is FOR). Documented fallback: if
    `model_version` matches what's currently in feature_registry.json, load
    straight from ml/artifacts/models/*.pkl -- keeps this usable in
    environments that haven't run `pipelines.train` yet (fresh clones, CI)."""
    name = "sentinel-cold" if is_cold else "sentinel-warm"
    try:
        mlflow.set_tracking_uri(f"file:{MLRUNS_DIR.as_posix()}")
        client = MlflowClient()
        matches = [
            mv for mv in client.search_model_versions(f"name='{name}'")
            if mv.tags.get("model_version") == model_version
        ]
        if matches:
            mv = matches[0]
            model = mlflow.lightgbm.load_model(f"models:/{name}/{mv.version}")
            calibrator_dir = mlflow.artifacts.download_artifacts(
                run_id=mv.run_id, artifact_path=f"{'cold' if is_cold else 'warm'}_calibrator")
            calibrator_file = next(Path(calibrator_dir).glob("*.pkl"))
            calibrator = joblib.load(calibrator_file)
            return model, calibrator
    except Exception:
        pass

    with open(FEATURE_REGISTRY_PATH) as f:
        registry = json.load(f)
    current_version = registry["cold_model" if is_cold else "warm_model"]["model_version"]
    if model_version != current_version:
        raise ValueError(
            f"No registered {name} version tagged model_version={model_version!r}, and it doesn't "
            f"match the current on-disk model ({current_version!r}) either. Run `pipelines.train` "
            f"first, or pass the current model_version."
        )
    model_path = COLD_MODEL_PATH if is_cold else WARM_MODEL_PATH
    calibrator_path = COLD_CALIBRATOR_PATH if is_cold else WARM_CALIBRATOR_PATH
    return joblib.load(model_path), joblib.load(calibrator_path)


def _counterpart_version(model_version: str, want_cold: bool) -> str:
    """cold/warm models trained together share a date-sha suffix (e.g.
    'cold-20260816-9a6747c' / 'warm-20260816-9a6747c') -- `--model` accepts
    either one and this derives the other, since a backtest window's rows
    are scored by BOTH models (routed by is_cold) and both are needed."""
    if "-" not in model_version:
        return model_version
    _prefix, rest = model_version.split("-", 1)
    return f"{'cold' if want_cold else 'warm'}-{rest}"


def _decide_with_thresholds(calibrated_prob: float, is_cold: bool, thresholds: dict) -> tuple:
    tier_thresholds = thresholds["cold" if is_cold else "warm"]
    tier = _risk_tier(calibrated_prob, tier_thresholds)
    action = _action_for_tier(tier)
    return action, tier


def _snapshot_for_model(feature_snapshot: dict) -> dict:
    """Postgres JSONB round-trips NaN as JSON `null` -> Python `None` (see
    decisionlog.writer._json_safe, Phase 5's NaN fix) -- reversed here before
    feeding a stored snapshot back into a model, which expects float('nan')
    for a missing numeric feature, never None (LightGBM rejects an `object`-
    dtype column). `amount_roundness` is the one registry feature that's a
    genuine string and never NaN-valued, so it's excluded from the reversal."""
    return {k: (float("nan") if v is None and k != "amount_roundness" else v)
            for k, v in feature_snapshot.items()}


def _rescore_row(row: dict, cold_model, cold_cal, warm_model, warm_cal, thresholds: dict) -> dict:
    is_cold = row["is_cold"]
    model, calibrator = (cold_model, cold_cal) if is_cold else (warm_model, warm_cal)
    feature_columns = COLD_FEATURES if is_cold else ALL_FEATURES

    X = vector_to_frame([_snapshot_for_model(row["feature_snapshot"])], feature_columns)
    raw = float(model.predict_proba(X)[:, 1][0])
    calibrated = float(apply_calibrator(calibrator, np.array([raw]))[0])
    action, tier = _decide_with_thresholds(calibrated, is_cold, thresholds)

    return {"txn_id": row["txn_id"], "raw_score": raw, "risk_score": calibrated,
            "action": action, "risk_tier": tier, "is_cold": is_cold}


def _cost_for_action(action: str, is_fraud: bool, amount: float) -> float:
    if action == "ALLOW":
        return FN_COST(amount) if is_fraud else 0.0
    if action == "WARN":
        return -TP_BENEFIT(amount) if is_fraud else FP_COST_FRICTION
    if action == "REVIEW":
        return -TP_BENEFIT(amount) if is_fraud else FP_COST_REVIEW
    return -TP_BENEFIT(amount) if is_fraud else FP_COST_BLOCK  # BLOCK


def _summarize(rows: list, actions: list, amounts: list, labels: list) -> dict:
    """`labels` may contain None for rows with unknown ground truth --
    excluded from fraud-dependent metrics, counted separately."""
    n = len(rows)
    known = [i for i in range(n) if labels[i] is not None]
    n_unknown = n - len(known)

    total_cost = sum(_cost_for_action(actions[i], bool(labels[i]), amounts[i]) for i in known)
    fraud_value = sum(amounts[i] for i in known if labels[i])
    caught_value = sum(amounts[i] for i in known if labels[i] and actions[i] != "ALLOW")
    n_fraud = sum(1 for i in known if labels[i])
    n_caught = sum(1 for i in known if labels[i] and actions[i] != "ALLOW")
    n_alerts = sum(1 for i in known if actions[i] != "ALLOW")
    n_false_positives = sum(1 for i in known if not labels[i] and actions[i] != "ALLOW")
    action_counts = {a: actions.count(a) for a in ("ALLOW", "WARN", "REVIEW", "BLOCK")}

    return {
        "n_rows": n,
        "n_excluded_unknown_label": n_unknown,
        "action_mix": action_counts,
        "n_fraud_known": n_fraud,
        "n_caught": n_caught,
        "n_false_positives": n_false_positives,
        "fraud_value_total": fraud_value,
        "fraud_value_caught": caught_value,
        "amount_weighted_recall": (caught_value / fraud_value) if fraud_value else None,
        "precision": (n_caught / n_alerts) if n_alerts else None,
        "expected_cost": total_cost,
    }


def _threshold_vs_net_benefit_curve(rows, cold_model, cold_cal, warm_model, warm_cal, base_thresholds,
                                     amounts, labels, is_cold_flags):
    """Sweeps t_review (warm model only, holding t_warn/t_block's RELATIVE
    gaps from the base thresholds) to trace the net-benefit tradeoff --
    the README headline curve."""
    known = [i for i in range(len(rows)) if labels[i] is not None]
    if not known:
        return []

    raw_scores = {}
    for i in known:
        model, cal = (cold_model, cold_cal) if is_cold_flags[i] else (warm_model, warm_cal)
        feature_columns = COLD_FEATURES if is_cold_flags[i] else ALL_FEATURES
        X = vector_to_frame([_snapshot_for_model(rows[i]["feature_snapshot"])], feature_columns)
        raw = float(model.predict_proba(X)[:, 1][0])
        raw_scores[i] = float(apply_calibrator(cal, np.array([raw]))[0])

    curve = []
    for t_review in np.arange(0.01, 1.0, 0.02):
        candidate = {
            "warm": {"t_warn": min(base_thresholds["warm"]["t_warn"], t_review),
                     "t_review": float(t_review),
                     "t_block": max(base_thresholds["warm"]["t_block"], t_review)},
            "cold": base_thresholds["cold"],
        }
        cost = 0.0
        for i in known:
            tier_thresholds = candidate["cold" if is_cold_flags[i] else "warm"]
            tier = _risk_tier(raw_scores[i], tier_thresholds)
            action = _action_for_tier(tier)
            cost += _cost_for_action(action, bool(labels[i]), amounts[i])
        curve.append({"t_review": float(t_review), "expected_cost": cost, "net_benefit": -cost})
    return curve


def run_backtest(date_from: str, date_to: str, model_version: str, thresholds_path: Path) -> dict:
    with open(thresholds_path) as f:
        thresholds = json.load(f)

    cold_version = _counterpart_version(model_version, want_cold=True)
    warm_version = _counterpart_version(model_version, want_cold=False)

    log = DecisionLog(DSN)
    rows_raw = log._conn.execute(
        "SELECT txn_id, event, feature_snapshot, risk_score, action, is_cold, model_version, "
        "thresholds_version FROM decisions WHERE scored_at >= %s AND scored_at < %s "
        "AND model_version IN (%s, %s) ORDER BY scored_at",
        (date_from, date_to, cold_version, warm_version),
    ).fetchall()
    columns = ("txn_id", "event", "feature_snapshot", "risk_score", "action", "is_cold",
               "model_version", "thresholds_version")
    rows = [dict(zip(columns, r)) for r in rows_raw]

    if not rows:
        raise ValueError(f"No decisions found for model_version={model_version!r} in [{date_from}, {date_to})")

    cold_model, cold_cal = _load_model_and_calibrator(cold_version, is_cold=True)
    warm_model, warm_cal = _load_model_and_calibrator(warm_version, is_cold=False)

    amounts = [float(r["event"].get("amount", 0.0)) for r in rows]
    labels = [r["event"]["label_is_fraud"] if "label_is_fraud" in r["event"] else None for r in rows]
    is_cold_flags = [r["is_cold"] for r in rows]

    # Scenario A: the decisions actually logged.
    original_actions = [r["action"] for r in rows]
    summary_a = _summarize(rows, original_actions, amounts, labels)

    # Scenario B: re-score every row's stored snapshot through the given model/thresholds.
    rescored = [_rescore_row(r, cold_model, cold_cal, warm_model, warm_cal, thresholds) for r in rows]
    new_actions = [r["action"] for r in rescored]
    summary_b = _summarize(rows, new_actions, amounts, labels)

    curve = _threshold_vs_net_benefit_curve(
        rows, cold_model, cold_cal, warm_model, warm_cal, thresholds, amounts, labels, is_cold_flags)

    net_benefit_delta = (-summary_b["expected_cost"]) - (-summary_a["expected_cost"])

    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "window": {"from": date_from, "to": date_to},
        "model_version": model_version,
        "thresholds_path": str(thresholds_path),
        "scenario_a_original": summary_a,
        "scenario_b_rescored": summary_b,
        "net_benefit_delta": net_benefit_delta,
        "threshold_vs_net_benefit_curve": curve,
    }

    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    out_path = BACKTEST_DIR / f"backtest_{int(time.time())}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    result["_written_to"] = str(out_path)
    return result


def _fmt(v):
    if isinstance(v, float):
        return f"{v:,.4f}" if abs(v) < 1000 else f"{v:,.2f}"
    return str(v)


def _print_report(result: dict):
    a, b = result["scenario_a_original"], result["scenario_b_rescored"]
    print(f"\nBacktest window: {result['window']['from']} .. {result['window']['to']}  "
          f"model={result['model_version']}")
    print(f"Rows: {a['n_rows']} ({a['n_excluded_unknown_label']} excluded from fraud metrics -- no ground truth)")
    print(f"\nAction mix -- A (original): {a['action_mix']}")
    print(f"Action mix -- B (rescored): {b['action_mix']}")
    print(f"\n{'metric':<28}{'A (original)':>20}{'B (rescored)':>20}")
    for key in ("n_caught", "n_false_positives", "fraud_value_caught",
                "amount_weighted_recall", "precision", "expected_cost"):
        print(f"{key:<28}{_fmt(a[key]):>20}{_fmt(b[key]):>20}")
    print(f"\nNet benefit delta (B vs A): {result['net_benefit_delta']:,.2f}")
    print("\nThreshold (t_review) vs net benefit:")
    for point in result["threshold_vs_net_benefit_curve"][::5]:
        print(f"  t_review={point['t_review']:.2f}  net_benefit={point['net_benefit']:,.2f}")
    print(f"\nWrote {result['_written_to']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Re-score a historical decision window from stored feature snapshots.")
    parser.add_argument("--from", dest="date_from", required=True)
    parser.add_argument("--to", dest="date_to", required=True)
    parser.add_argument("--model", dest="model_version", required=True)
    parser.add_argument("--thresholds", dest="thresholds_path", required=True, type=Path)
    args = parser.parse_args(argv)

    result = run_backtest(args.date_from, args.date_to, args.model_version, args.thresholds_path)
    _print_report(result)
    return result


if __name__ == "__main__":
    main()
