"""Threshold optimization: finds (t_warn, t_review, t_block) per model that
minimizes expected cost on the VALIDATION split (never test), subject to a
REVIEW+BLOCK capacity cap.

Sequential (greedy), not a joint 3D grid: an exhaustive joint sweep at
0.005 steps is ~200^3 ~= 8M candidate triples x ~50k validation events --
hours of runtime. Instead, three single-threshold passes, ~200 candidates
each (~600 total):

  1. Sweep t_review alone, treating everything >= t_review as one combined
     REVIEW+BLOCK action (costed as REVIEW). Fix the best value.
  2. Sweep t_block over (t_review, 1] with t_review fixed, now scoring
     REVIEW vs BLOCK separately. Fix the best value.
  3. Sweep t_warn over [0, t_review) with both fixed, scoring ALLOW vs WARN.

This finds a greedy SEQUENTIAL optimum, not an exhaustive JOINT one. Accepted
because the cost surface is monotone in each threshold given the others held
fixed (moving t_review up always trades some FN cost for less FP-review
cost in one direction), and full joint search is intractable at this step
size. Each pass is vectorized across all candidates at once via numpy
broadcasting -- no per-event Python loop.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ml.src.policy.cost_matrix import (  # noqa: E402
    FN_COST,
    FP_COST_BLOCK,
    FP_COST_FRICTION,
    FP_COST_REVIEW,
    TP_BENEFIT,
)
from ml.src.train import build_feature_dataframe  # noqa: E402
from ml.src.training.lightgbm_pipeline import apply_calibrator  # noqa: E402
from ml.src.training.temporal_split import split_by_day  # noqa: E402
from ml.src.data.load_synthetic import load_synthetic_events  # noqa: E402
from feature_lib.registry import ALL_FEATURES, COLD_FEATURES  # noqa: E402
from ml.src.utils.paths import (  # noqa: E402
    COLD_CALIBRATOR_PATH,
    COLD_MODEL_PATH,
    POLICY_DIR,
    THRESHOLD_CURVE_PATH,
    THRESHOLDS_PATH,
    WARM_CALIBRATOR_PATH,
    WARM_MODEL_PATH,
)

MAX_REVIEW_BLOCK_RATE = 0.02
STEP = 0.005
CANDIDATES = np.arange(0.0, 1.0 + STEP, STEP)

# Applied only if the sequential sweep doesn't naturally produce cold >= warm
# at every tier (a small cold sample can pick a noisy optimum). Documented,
# not silent.
COLD_STRICTNESS_MULTIPLIER = 1.15


def _sweep_t_review(p, y, amount, candidates=CANDIDATES, max_rate=MAX_REVIEW_BLOCK_RATE):
    """Pass 1: below candidate = ALLOW, above = combined REVIEW+BLOCK
    (costed as REVIEW). Returns (best_t_review, best_cost, sweep_curve)."""
    t = candidates[:, None]          # (K,1)
    pp = p[None, :]                   # (1,N)
    yy = y[None, :]
    aa = amount[None, :]

    above = pp >= t                    # (K,N)
    below = ~above
    fraud = yy == 1

    n_above = above.sum(axis=1)
    valid = (n_above / p.shape[0]) <= max_rate

    cost_below = np.where(below & fraud, FN_COST(aa), 0.0).sum(axis=1)
    cost_above = np.where(above & fraud, -TP_BENEFIT(aa), np.where(above & ~fraud, FP_COST_REVIEW, 0.0)).sum(axis=1)
    total = cost_below + cost_above

    if valid.any():
        masked = np.where(valid, total, np.inf)
        best_idx = int(np.argmin(masked))
    else:
        # The cap is infeasible at this sample size/probability distribution
        # (e.g. cold's tiny validation set can produce near-step-function
        # calibrated probabilities where no threshold keeps REVIEW+BLOCK
        # under 2%). Falling back to argmin over an all-inf array would
        # silently pick candidate[0] (t=0.0 -- the worst possible choice,
        # flagging everything). Instead, fall back to whichever candidate(s)
        # achieve the smallest actual n_above, and pick the cheapest among
        # those -- the least-bad honest choice, not a silent default.
        min_n_above = n_above.min()
        near_best = n_above == min_n_above
        masked = np.where(near_best, total, np.inf)
        best_idx = int(np.argmin(masked))
        print(f"  WARNING: no threshold keeps REVIEW+BLOCK <= {max_rate:.0%} of "
              f"{p.shape[0]} events -- falling back to the threshold with the smallest "
              f"achievable rate ({min_n_above}/{p.shape[0]} = {min_n_above / p.shape[0]:.1%}).")

    curve = list(zip(candidates.tolist(), total.tolist(), valid.tolist()))
    return float(candidates[best_idx]), float(total[best_idx]), curve


def _sweep_t_block(p, y, amount, t_review, candidates=CANDIDATES):
    """Pass 2: t_review fixed. below t_review = ALLOW, [t_review, t_block) =
    REVIEW, >= t_block = BLOCK. Candidates restricted to (t_review, 1]."""
    valid_candidates = candidates[candidates > t_review]
    if len(valid_candidates) == 0:
        valid_candidates = np.array([1.0])

    t = valid_candidates[:, None]
    pp = p[None, :]
    yy = y[None, :]
    aa = amount[None, :]
    fraud = yy == 1

    allow = pp < t_review
    block = pp >= t
    review = (~allow) & (~block)

    cost_allow = np.where(allow & fraud, FN_COST(aa), 0.0).sum(axis=1)
    cost_review = np.where(review & fraud, -TP_BENEFIT(aa), np.where(review & ~fraud, FP_COST_REVIEW, 0.0)).sum(axis=1)
    cost_block = np.where(block & fraud, -TP_BENEFIT(aa), np.where(block & ~fraud, FP_COST_BLOCK, 0.0)).sum(axis=1)
    total = cost_allow + cost_review + cost_block

    best_idx = int(np.argmin(total))
    curve = list(zip(valid_candidates.tolist(), total.tolist()))
    return float(valid_candidates[best_idx]), float(total[best_idx]), curve


def _sweep_t_warn(p, y, amount, t_review, t_block, candidates=CANDIDATES):
    """Pass 3: t_review, t_block fixed. below t_warn = ALLOW,
    [t_warn, t_review) = WARN, [t_review, t_block) = REVIEW, >= t_block = BLOCK.
    Candidates restricted to [0, t_review)."""
    valid_candidates = candidates[candidates < t_review]
    if len(valid_candidates) == 0:
        valid_candidates = np.array([0.0])

    t = valid_candidates[:, None]
    pp = p[None, :]
    yy = y[None, :]
    aa = amount[None, :]
    fraud = yy == 1

    allow = pp < t
    warn = (pp >= t) & (pp < t_review)
    review = (pp >= t_review) & (pp < t_block)
    block = pp >= t_block

    cost_allow = np.where(allow & fraud, FN_COST(aa), 0.0).sum(axis=1)
    cost_warn = np.where(warn & fraud, -TP_BENEFIT(aa), np.where(warn & ~fraud, FP_COST_FRICTION, 0.0)).sum(axis=1)
    cost_review = np.where(review & fraud, -TP_BENEFIT(aa), np.where(review & ~fraud, FP_COST_REVIEW, 0.0)).sum(axis=1)
    cost_block = np.where(block & fraud, -TP_BENEFIT(aa), np.where(block & ~fraud, FP_COST_BLOCK, 0.0)).sum(axis=1)
    total = cost_allow + cost_warn + cost_review + cost_block

    best_idx = int(np.argmin(total))
    curve = list(zip(valid_candidates.tolist(), total.tolist()))
    return float(valid_candidates[best_idx]), float(total[best_idx]), curve


def optimize_thresholds(p, y, amount):
    """Runs all three passes in sequence. Returns a dict with the chosen
    thresholds, the expected cost at that point, and the full sweep curves
    (for threshold_curve.json)."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    amount = np.asarray(amount, dtype=float)

    t_review, _, review_curve = _sweep_t_review(p, y, amount)
    t_block, _, block_curve = _sweep_t_block(p, y, amount, t_review)
    t_warn, final_cost, warn_curve = _sweep_t_warn(p, y, amount, t_review, t_block)

    return {
        "t_warn": t_warn,
        "t_review": t_review,
        "t_block": t_block,
        "expected_cost": final_cost,
        "n_events": len(p),
        "curves": {"t_review_pass": review_curve, "t_block_pass": block_curve, "t_warn_pass": warn_curve},
    }


def naive_cutoff_cost(p, y, amount, cutoff=0.5):
    """Expected cost of a single naive 0.5 cutoff (ALLOW below, BLOCK
    above) -- the headline comparison point."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    amount = np.asarray(amount, dtype=float)
    fraud = y == 1
    above = p >= cutoff
    below = ~above
    cost = (
        np.where(below & fraud, FN_COST(amount), 0.0).sum()
        + np.where(above & fraud, -TP_BENEFIT(amount), 0.0).sum()
        + np.where(above & ~fraud, FP_COST_BLOCK, 0.0).sum()
    )
    return float(cost)


def _rate_metrics(p, y, amount, thresholds):
    """Alert rate / precision / recall / amount-weighted recall / cost
    breakdown at a chosen threshold triple -- for threshold_curve.json."""
    y = np.asarray(y, dtype=int)
    amount = np.asarray(amount, dtype=float)
    fraud = y == 1

    allow = p < thresholds["t_warn"]
    warn = (p >= thresholds["t_warn"]) & (p < thresholds["t_review"])
    review = (p >= thresholds["t_review"]) & (p < thresholds["t_block"])
    block = p >= thresholds["t_block"]
    alerted = ~allow

    total_fraud_value = amount[fraud].sum()
    caught_value = amount[fraud & alerted].sum()

    n = len(p)
    n_alerts = int(alerted.sum())
    n_caught = int((fraud & alerted).sum())

    return {
        "alert_rate": n_alerts / n,
        "review_block_rate": float((review | block).sum() / n),
        "precision": (n_caught / n_alerts) if n_alerts else 0.0,
        "recall": (n_caught / fraud.sum()) if fraud.sum() else 0.0,
        "amount_weighted_recall": (caught_value / total_fraud_value) if total_fraud_value else 0.0,
        "expected_cost": (
            np.where(allow & fraud, FN_COST(amount), 0.0).sum()
            + np.where(warn & fraud, -TP_BENEFIT(amount), np.where(warn & ~fraud, FP_COST_FRICTION, 0.0)).sum()
            + np.where(review & fraud, -TP_BENEFIT(amount), np.where(review & ~fraud, FP_COST_REVIEW, 0.0)).sum()
            + np.where(block & fraud, -TP_BENEFIT(amount), np.where(block & ~fraud, FP_COST_BLOCK, 0.0)).sum()
        ),
        "prevented_loss": float(amount[fraud & alerted].sum()),
        "friction_cost": float(warn[~fraud].sum() * FP_COST_FRICTION
                                + review[~fraud].sum() * FP_COST_REVIEW
                                + block[~fraud].sum() * FP_COST_BLOCK),
    }


def _build_val_probabilities():
    events = load_synthetic_events()
    df = build_feature_dataframe(events)
    sim_start = df["timestamp"].min()
    _train_df, val_df, _test_df = split_by_day(df, sim_start)

    cold_val = val_df[val_df["is_cold"]]
    warm_val = val_df[~val_df["is_cold"]]

    cold_model = joblib.load(COLD_MODEL_PATH)
    warm_model = joblib.load(WARM_MODEL_PATH)
    cold_calibrator = joblib.load(COLD_CALIBRATOR_PATH)
    warm_calibrator = joblib.load(WARM_CALIBRATOR_PATH)

    cold_proba = apply_calibrator(cold_calibrator, cold_model.predict_proba(cold_val[COLD_FEATURES])[:, 1])
    warm_proba = apply_calibrator(warm_calibrator, warm_model.predict_proba(warm_val[ALL_FEATURES])[:, 1])

    return {
        "cold": (cold_proba, cold_val["label_is_fraud"].to_numpy(), cold_val["amount"].to_numpy()),
        "warm": (warm_proba, warm_val["label_is_fraud"].to_numpy(), warm_val["amount"].to_numpy()),
    }


def main():
    data = _build_val_probabilities()

    results = {}
    for model_name, (p, y, amount) in data.items():
        result = optimize_thresholds(p, y, amount)
        results[model_name] = result
        print(f"{model_name}: t_warn={result['t_warn']:.3f} t_review={result['t_review']:.3f} "
              f"t_block={result['t_block']:.3f} expected_cost={result['expected_cost']:.2f} "
              f"(n={result['n_events']})")

    # Cold-strictness: assert cold >= warm at every tier; apply a documented
    # multiplier if the sequential sweep didn't produce it naturally.
    applied_multiplier = False
    for tier in ("t_warn", "t_review", "t_block"):
        if results["cold"][tier] < results["warm"][tier]:
            applied_multiplier = True
    if applied_multiplier:
        print(f"\nCold thresholds were not strictly >= warm at every tier after the "
              f"sequential sweep -- applying documented COLD_STRICTNESS_MULTIPLIER="
              f"{COLD_STRICTNESS_MULTIPLIER} to cold thresholds.")
        for tier in ("t_warn", "t_review", "t_block"):
            warm_val_t = results["warm"][tier]
            cold_val_t = results["cold"][tier]
            new_val = max(cold_val_t * COLD_STRICTNESS_MULTIPLIER, warm_val_t)
            results["cold"][tier] = min(1.0, new_val)
        # Re-order in case the multiplier pushed a lower tier above a higher one.
        ordered = sorted((results["cold"]["t_warn"], results["cold"]["t_review"], results["cold"]["t_block"]))
        results["cold"]["t_warn"], results["cold"]["t_review"], results["cold"]["t_block"] = ordered

    for tier in ("t_warn", "t_review", "t_block"):
        assert results["cold"][tier] >= results["warm"][tier], (
            f"cold {tier}={results['cold'][tier]} is not >= warm {tier}={results['warm'][tier]}"
        )

    # metrics/threshold_curve.json: rate metrics at each t_review pass
    # candidate (representative operating points across the sweep), per model.
    threshold_curve = {}
    for model_name, (p, y, amount) in data.items():
        chosen = results[model_name]
        curve_points = []
        for t_review_candidate, _cost, valid in chosen["curves"]["t_review_pass"][::4]:  # thin for file size
            if not valid:
                continue
            probe_thresholds = {
                "t_warn": min(chosen["t_warn"], t_review_candidate),
                "t_review": t_review_candidate,
                "t_block": max(chosen["t_block"], t_review_candidate),
            }
            curve_points.append({"t_review": t_review_candidate, **_rate_metrics(p, y, amount, probe_thresholds)})
        threshold_curve[model_name] = {
            "chosen": _rate_metrics(p, y, amount, results[model_name]),
            "sweep": curve_points,
        }

    POLICY_DIR.mkdir(parents=True, exist_ok=True)
    with open(THRESHOLDS_PATH, "w") as f:
        json.dump({
            "cold": {k: v for k, v in results["cold"].items() if k != "curves"},
            "warm": {k: v for k, v in results["warm"].items() if k != "curves"},
            "max_review_block_rate": MAX_REVIEW_BLOCK_RATE,
            "cold_strictness_multiplier_applied": applied_multiplier,
        }, f, indent=2)

    with open(THRESHOLD_CURVE_PATH, "w") as f:
        json.dump(threshold_curve, f, indent=2)

    print(f"\nWrote {THRESHOLDS_PATH}")
    print(f"Wrote {THRESHOLD_CURVE_PATH}")
    return results


if __name__ == "__main__":
    main()
