"""Drift monitoring -- compares a recent window of `decisions` against the
offline training/validation reference distributions along three EXPLICITLY
SEPARATE axes, because each has a different cause and a different remedy:

  - FEATURE drift (PSI per feature): has the input population changed vs
    what the model was trained on? Remedy: retrain on new data.
  - SCORE drift (KS test on risk_score): is the model's opinion distribution
    shifting vs validation time (could be feature drift propagating through,
    or just a different mix of cases)? Remedy: re-tune thresholds.
  - CONCEPT drift (calibration on reviewed cases only): among cases an
    analyst has actually labeled, does risk_score still track the true
    fraud rate? Remedy: get more labels / retrain. Carries the SAME
    selective-labelling caveat as the review queue -- see DESIGN.md.

    python -m pipelines.drift_check [--recent-days 30]
    make drift

Writes ml/artifacts/drift/drift_<ts>.json.

Import-order note (load-bearing, same issue as pipelines/backtest.py and
ml/src/policy/reason_codes.py): `import lightgbm` MUST be the literal first
import in this file, before anything that transitively imports pandas
(ml.src.train does), or LightGBM's ctypes bridge crashes with a native
access-violation the first time its C API is touched.
"""
import lightgbm  # noqa: F401,E402,I001 -- MUST be imported before pandas, see module docstring
import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from scipy.stats import ks_2samp  # noqa: E402

from decisionlog import DecisionLog  # noqa: E402
from feature_lib.registry import ALL_FEATURES  # noqa: E402
from ml.src.data.load_synthetic import load_synthetic_events  # noqa: E402
from ml.src.policy.thresholds import _build_val_probabilities  # noqa: E402
from ml.src.train import build_feature_dataframe  # noqa: E402
from ml.src.training.temporal_split import split_by_day  # noqa: E402
from ml.src.utils.paths import ARTIFACTS_DIR  # noqa: E402

DRIFT_DIR = ARTIFACTS_DIR / "drift"
# Same DATABASE_URL convention config/settings.py, docker-compose.yml, and
# pipelines/backtest.py use; falls back to the local-Docker default for ad
# hoc runs outside a container.
DSN = os.environ.get("DATABASE_URL", "postgresql://sentinel:sentinel@localhost:5432/sentinel")

PSI_MODERATE = 0.1
PSI_SIGNIFICANT = 0.25
N_PSI_BINS = 10
MIN_REVIEWED_FOR_CALIBRATION = 5

# The one registry feature that's a genuine string category, never NaN-valued
# -- same list ml/src/train.py's CATEGORICAL_FEATURES uses for LightGBM's
# categorical_feature=... argument.
CATEGORICAL_FEATURES = ["amount_roundness"]


def _training_reference_df():
    """Rebuilds the training split's feature distribution from the raw
    synthetic events -- same functions ml/src/train.py itself uses, so this
    is guaranteed to match what the currently-deployed models actually saw.
    Recomputes features for the whole dataset; not cheap, but drift_check is
    a periodic batch job, not a hot path."""
    events = load_synthetic_events()
    df = build_feature_dataframe(events)
    sim_start = df["timestamp"].min()
    train_df, _val_df, _test_df = split_by_day(df, sim_start)
    return train_df


def _snapshot_value(feature_snapshot: dict, feature: str):
    v = feature_snapshot.get(feature)
    return float("nan") if v is None else v


def _psi(reference: np.ndarray, current: np.ndarray, n_bins: int = N_PSI_BINS) -> float:
    """Population Stability Index between a reference and current sample of
    one feature, quantile-binned on the reference distribution. NaNs are
    treated as their own bin (missingness rate is itself a signal worth
    catching, not something to silently drop)."""
    ref_nan = np.isnan(reference)
    cur_nan = np.isnan(current)
    ref_valid, cur_valid = reference[~ref_nan], current[~cur_nan]

    if len(ref_valid) == 0:
        return 0.0

    edges = np.unique(np.quantile(ref_valid, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        edges = np.array([ref_valid.min() - 1e-9, ref_valid.max() + 1e-9])

    ref_counts, _ = np.histogram(ref_valid, bins=edges)
    cur_counts, _ = np.histogram(cur_valid, bins=edges)
    ref_counts = np.append(ref_counts, ref_nan.sum())
    cur_counts = np.append(cur_counts, cur_nan.sum())

    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)

    eps = 1e-4
    ref_pct = np.where(ref_pct == 0, eps, ref_pct)
    cur_pct = np.where(cur_pct == 0, eps, cur_pct)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _psi_categorical(reference: np.ndarray, current: np.ndarray) -> float:
    """PSI variant for `amount_roundness`, the one registry feature that's a
    genuine string category (not NaN-valued) rather than a numeric column --
    quantile binning doesn't apply, so each observed category is its own bin."""
    categories = sorted(set(reference) | set(current))
    ref_counts = np.array([np.sum(reference == c) for c in categories], dtype=float)
    cur_counts = np.array([np.sum(current == c) for c in categories], dtype=float)

    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)

    eps = 1e-4
    ref_pct = np.where(ref_pct == 0, eps, ref_pct)
    cur_pct = np.where(cur_pct == 0, eps, cur_pct)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _severity(psi: float) -> str:
    if psi > PSI_SIGNIFICANT:
        return "significant"
    if psi > PSI_MODERATE:
        return "moderate"
    return "stable"


def _feature_drift(recent_rows: list, train_df) -> dict:
    psi_by_feature = {}
    for feature in ALL_FEATURES:
        if feature in CATEGORICAL_FEATURES:
            reference = train_df[feature].astype(str).to_numpy()
            current = np.array(
                [str(_snapshot_value(r["feature_snapshot"], feature)) for r in recent_rows],
            )
            psi_by_feature[feature] = _psi_categorical(reference, current)
            continue

        reference = train_df[feature].to_numpy(dtype=float)
        current = np.array(
            [_snapshot_value(r["feature_snapshot"], feature) for r in recent_rows], dtype=float,
        )
        psi_by_feature[feature] = _psi(reference, current)

    return {
        "psi_by_feature": psi_by_feature,
        "moderate": sorted([f for f, p in psi_by_feature.items() if PSI_MODERATE < p <= PSI_SIGNIFICANT]),
        "significant": sorted([f for f, p in psi_by_feature.items() if p > PSI_SIGNIFICANT]),
        "thresholds": {"moderate": PSI_MODERATE, "significant": PSI_SIGNIFICANT},
        "cold_start_caveat": (
            "History-dependent features (payer/payee txn counts, ages, rates) will show "
            "large PSI whenever `recent_rows` comes from a worker replay that started "
            "against an EMPTY history store -- e.g. the Phase 6 8k-event capped replay. "
            "Training's build_feature_dataframe warms one continuous store across all "
            "90 days; a fresh replay has no such warm-up, so counts/ages/rates read "
            "artificially low/zero at first. This is a replay-seeding artifact, not "
            "evidence of real population drift, until the store has been running long "
            "enough to warm back up. See PROGRESS.md's Phase 6 section."
        ),
    }


def _score_drift(recent_rows: list) -> dict:
    """KS test: recent decisions.risk_score vs the calibrated validation
    probabilities that were used to CHOOSE the currently deployed
    thresholds. This answers 'is the model producing a different opinion
    distribution than it did at threshold-tuning time', regardless of
    whether ground truth is known for the recent rows."""
    val_data = _build_val_probabilities()
    val_scores = np.concatenate([val_data["cold"][0], val_data["warm"][0]])
    recent_scores = np.array([r["risk_score"] for r in recent_rows], dtype=float)

    statistic, pvalue = ks_2samp(val_scores, recent_scores)
    return {
        "n_validation": len(val_scores),
        "n_recent": len(recent_scores),
        "ks_statistic": float(statistic),
        "ks_pvalue": float(pvalue),
        "significant_drift": bool(pvalue < 0.05),
    }


def _calibration_drift_on_reviewed(conn, date_from: str) -> dict:
    """Concept drift among REVIEWED cases only -- selective-labelling bias
    applies here exactly as it does to the review queue's precision figure
    (see DESIGN.md): only alerted-and-then-reviewed transactions appear in
    this comparison, so it says nothing about calibration in the ALLOW tier
    or among alerted-but-not-yet-reviewed cases."""
    rows = conn.execute(
        "SELECT d.risk_score, rl.disposition FROM decisions d "
        "JOIN users_reviewlabel rl ON rl.decision_id = d.id "
        "WHERE d.scored_at >= %s AND rl.disposition != 'UNCLEAR'",
        (date_from,),
    ).fetchall()

    if len(rows) < MIN_REVIEWED_FOR_CALIBRATION:
        return {
            "skipped": True,
            "reason": f"only {len(rows)} labeled case(s) with a definite disposition "
                      f"(need >= {MIN_REVIEWED_FOR_CALIBRATION}) -- reported calibration "
                      f"drift would be noise, not signal.",
            "caveat": "Selective-labelling bias: even once populated, this only covers "
                      "alerted-and-reviewed cases, never the ALLOW tier. See DESIGN.md.",
        }

    scores = np.array([r[0] for r in rows], dtype=float)
    labels = np.array([1 if r[1] == "CONFIRMED_FRAUD" else 0 for r in rows], dtype=float)

    edges = np.unique(np.quantile(scores, np.linspace(0, 1, min(5, len(scores)) + 1)))
    bin_idx = np.clip(np.digitize(scores, edges[1:-1]), 0, len(edges) - 2)

    bins = []
    for b in range(len(edges) - 1):
        mask = bin_idx == b
        if not mask.any():
            continue
        bins.append({
            "n": int(mask.sum()),
            "mean_predicted_score": float(scores[mask].mean()),
            "empirical_fraud_rate": float(labels[mask].mean()),
        })

    mean_abs_calibration_error = float(
        np.mean([abs(b["mean_predicted_score"] - b["empirical_fraud_rate"]) for b in bins]),
    ) if bins else None

    return {
        "skipped": False,
        "n_reviewed_with_definite_disposition": len(rows),
        "bins": bins,
        "mean_abs_calibration_error": mean_abs_calibration_error,
        "caveat": "Selective-labelling bias: only alerted-and-reviewed cases are represented "
                  "here, never the ALLOW tier. See DESIGN.md.",
    }


def _action_mix_over_time(conn, date_from: str) -> list:
    rows = conn.execute(
        "SELECT date_trunc('day', scored_at) AS day, action, count(*) AS n "
        "FROM decisions WHERE scored_at >= %s GROUP BY 1, 2 ORDER BY 1, 2",
        (date_from,),
    ).fetchall()
    return [{"day": str(r[0]), "action": r[1], "n": r[2]} for r in rows]


def run_drift_check(recent_days: int = 30) -> dict:
    date_from = (datetime.now(timezone.utc) - timedelta(days=recent_days)).isoformat()

    log = DecisionLog(DSN)
    rows_raw = log._conn.execute(
        "SELECT txn_id, feature_snapshot, risk_score, action, is_cold FROM decisions "
        "WHERE scored_at >= %s ORDER BY scored_at",
        (date_from,),
    ).fetchall()
    columns = ("txn_id", "feature_snapshot", "risk_score", "action", "is_cold")
    recent_rows = [dict(zip(columns, r)) for r in rows_raw]

    if not recent_rows:
        raise ValueError(
            f"No decisions found in the last {recent_days} day(s) -- nothing to check drift "
            f"against. Widen --recent-days or replay more traffic first.",
        )

    train_df = _training_reference_df()

    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "window": {"recent_days": recent_days, "from": date_from},
        "n_recent_decisions": len(recent_rows),
        "feature_drift": _feature_drift(recent_rows, train_df),
        "score_drift": _score_drift(recent_rows),
        "calibration_drift_on_reviewed_cases": _calibration_drift_on_reviewed(log._conn, date_from),
        "action_mix_over_time": _action_mix_over_time(log._conn, date_from),
    }

    DRIFT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DRIFT_DIR / f"drift_{int(time.time())}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    result["_written_to"] = str(out_path)
    return result


def _print_report(result: dict):
    fd, sd = result["feature_drift"], result["score_drift"]
    print(f"\nDrift check over last {result['window']['recent_days']} day(s) "
          f"({result['n_recent_decisions']} decisions)")

    print("\n-- FEATURE drift (PSI vs training distribution) --")
    if fd["significant"]:
        print(f"  SIGNIFICANT (>{PSI_SIGNIFICANT}): {fd['significant']}")
    if fd["moderate"]:
        print(f"  moderate ({PSI_MODERATE}-{PSI_SIGNIFICANT}): {fd['moderate']}")
    if not fd["significant"] and not fd["moderate"]:
        print("  All features stable.")
    top5 = sorted(fd["psi_by_feature"].items(), key=lambda kv: kv[1], reverse=True)[:5]
    print("  Top 5 by PSI:", [(f, round(p, 4)) for f, p in top5])
    print(f"  CAVEAT: {fd['cold_start_caveat']}")

    print("\n-- SCORE drift (KS test, recent risk_score vs validation-time calibrated probabilities) --")
    print(f"  KS statistic={sd['ks_statistic']:.4f}  p-value={sd['ks_pvalue']:.4g}  "
          f"significant_drift={sd['significant_drift']}")

    cd = result["calibration_drift_on_reviewed_cases"]
    print("\n-- CONCEPT drift (calibration on reviewed cases only -- selective-labelling bias applies) --")
    if cd["skipped"]:
        print(f"  Skipped: {cd['reason']}")
    else:
        print(f"  n={cd['n_reviewed_with_definite_disposition']}  "
              f"mean_abs_calibration_error={cd['mean_abs_calibration_error']:.4f}")
    print(f"  {cd['caveat']}")

    print("\n-- Action mix over time (first 10 rows) --")
    for row in result["action_mix_over_time"][:10]:
        print(f"  {row['day']}  {row['action']:<8}  n={row['n']}")

    print(f"\nWrote {result['_written_to']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check decisions for feature/score/concept drift.")
    parser.add_argument("--recent-days", dest="recent_days", type=int, default=30)
    args = parser.parse_args(argv)

    result = run_drift_check(args.recent_days)
    _print_report(result)
    return result


if __name__ == "__main__":
    main()
