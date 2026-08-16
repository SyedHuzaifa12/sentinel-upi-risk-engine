"""End-to-end demo of the Phase 3 decision policy against real test-set
events: features -> model -> calibration -> decide() -> reason_codes().

Stands in for live Django wiring (deferred to Phase 4, since the live form
can't supply feature_lib features -- see PROGRESS.md). This is how the new
cold/warm models + policy layer are actually exercised end-to-end today.

    python ml/src/policy/demo.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from feature_lib.registry import ALL_FEATURES, COLD_FEATURES  # noqa: E402
from ml.src.data.load_synthetic import load_synthetic_events  # noqa: E402
from ml.src.policy.decide import decide  # noqa: E402
from ml.src.policy.reason_codes import compute_reason_codes, measure_shap_latency_ms  # noqa: E402
from ml.src.policy.thresholds import naive_cutoff_cost  # noqa: E402
from ml.src.train import build_feature_dataframe  # noqa: E402
from ml.src.training.lightgbm_pipeline import apply_calibrator  # noqa: E402
from ml.src.training.temporal_split import split_by_day  # noqa: E402
from ml.src.utils.paths import (  # noqa: E402
    COLD_CALIBRATOR_PATH,
    COLD_MODEL_PATH,
    THRESHOLDS_PATH,
    WARM_CALIBRATOR_PATH,
    WARM_MODEL_PATH,
)


def main():
    with open(THRESHOLDS_PATH) as f:
        thresholds = json.load(f)

    print("Chosen thresholds per model:")
    for model_name in ("cold", "warm"):
        t = thresholds[model_name]
        print(f"  {model_name}: t_warn={t['t_warn']:.3f} t_review={t['t_review']:.3f} "
              f"t_block={t['t_block']:.3f}")

    print("\nLoading synthetic events and computing features (test split)...")
    events = load_synthetic_events()
    df = build_feature_dataframe(events)
    sim_start = df["timestamp"].min()
    _train_df, _val_df, test_df = split_by_day(df, sim_start)

    cold_test = test_df[test_df["is_cold"]]
    warm_test = test_df[~test_df["is_cold"]]

    cold_model = joblib.load(COLD_MODEL_PATH)
    warm_model = joblib.load(WARM_MODEL_PATH)
    cold_calibrator = joblib.load(COLD_CALIBRATOR_PATH)
    warm_calibrator = joblib.load(WARM_CALIBRATOR_PATH)

    cold_proba = apply_calibrator(cold_calibrator, cold_model.predict_proba(cold_test[COLD_FEATURES])[:, 1])
    warm_proba = apply_calibrator(warm_calibrator, warm_model.predict_proba(warm_test[ALL_FEATURES])[:, 1])

    all_proba = np.concatenate([cold_proba, warm_proba])
    all_y = np.concatenate([cold_test["label_is_fraud"].to_numpy(), warm_test["label_is_fraud"].to_numpy()])
    all_amount = np.concatenate([cold_test["amount"].to_numpy(), warm_test["amount"].to_numpy()])

    # Headline result: expected cost at the chosen operating point vs a naive 0.5 cutoff.
    def _decision_cost(proba, y, amount, is_cold_flags):
        from ml.src.policy.cost_matrix import FN_COST, FP_COST_BLOCK, FP_COST_FRICTION, FP_COST_REVIEW, TP_BENEFIT
        total = 0.0
        for p, label, amt, is_cold in zip(proba, y, amount, is_cold_flags):
            action = decide(p, is_cold, amt).action
            fraud = label == 1
            if action == "ALLOW":
                total += FN_COST(amt) if fraud else 0.0
            elif action == "WARN":
                total += -TP_BENEFIT(amt) if fraud else FP_COST_FRICTION
            elif action == "REVIEW":
                total += -TP_BENEFIT(amt) if fraud else FP_COST_REVIEW
            else:
                total += -TP_BENEFIT(amt) if fraud else FP_COST_BLOCK
        return total

    is_cold_flags = [True] * len(cold_proba) + [False] * len(warm_proba)
    chosen_cost = _decision_cost(all_proba, all_y, all_amount, is_cold_flags)
    naive_cost = naive_cutoff_cost(all_proba, all_y, all_amount, cutoff=0.5)

    print(f"\nExpected cost at chosen thresholds: {chosen_cost:,.2f}")
    print(f"Expected cost at naive 0.5 cutoff:   {naive_cost:,.2f}")
    print(f"(lower is better; negative = net benefit)")

    n_alerts = sum(1 for p, is_cold in zip(all_proba, is_cold_flags)
                   if decide(p, is_cold, 0).action != "ALLOW")
    n_caught = sum(1 for p, label, is_cold in zip(all_proba, all_y, is_cold_flags)
                   if label == 1 and decide(p, is_cold, 0).action != "ALLOW")
    n_fraud = int(all_y.sum())
    fraud_value = all_amount[all_y == 1].sum()
    caught_value = sum(amt for p, label, amt, is_cold in zip(all_proba, all_y, all_amount, is_cold_flags)
                        if label == 1 and decide(p, is_cold, amt).action != "ALLOW")

    print(f"\nAlert rate: {n_alerts / len(all_proba):.2%}")
    print(f"Precision: {(n_caught / n_alerts if n_alerts else 0):.2%}")
    print(f"Amount-weighted recall: {(caught_value / fraud_value if fraud_value else 0):.2%}")

    # Worked example: a real fraud event, escalated past ALLOW.
    warm_test = warm_test.reset_index(drop=True)
    fraud_mask = (warm_test["label_is_fraud"] == 1) & (warm_proba >= thresholds["warm"]["t_warn"])
    fraud_indices = warm_test.index[fraud_mask]
    if len(fraud_indices) > 0:
        idx = fraud_indices[0]
        row = warm_test.iloc[idx]
        proba = float(warm_proba[idx])
        feature_vector = row[ALL_FEATURES].to_dict()
        decision = decide(proba, is_cold=False, amount=row["amount"])
        codes = compute_reason_codes(feature_vector, is_cold=False)

        print("\n" + "=" * 70)
        print("WORKED EXAMPLE (real fraud event from the test set)")
        print("=" * 70)
        print(f"  txn_id: {row['txn_id']}  typology: {row['label_typology']}  amount: Rs {row['amount']:.2f}")
        print(f"  score: {decision.risk_score:.4f}  action: {decision.action}  tier: {decision.risk_tier}")
        print(f"  model_version: {decision.model_version}  thresholds_version: {decision.thresholds_version}")
        print("  reason codes:")
        for c in codes:
            print(f"    - {c['message']}  (shap={c['shap_contribution']:.4f})")
    else:
        print("\nNo fraud event in the test set crossed t_warn -- no worked example available.")

    # SHAP latency, measured honestly (never used to drop reason codes).
    sample = warm_test.iloc[:50][ALL_FEATURES].to_dict("records")
    latency = measure_shap_latency_ms(sample, [False] * len(sample))
    print(f"\nSHAP latency over {latency['n']} calls: p50={latency['p50_ms']:.1f}ms p99={latency['p99_ms']:.1f}ms "
          f"(budget: 20ms)")
    if latency["p99_ms"] > 20.0:
        print("  NOTE: p99 exceeds the 20ms budget. Reason codes are still generated in full -- "
              "never silently dropped -- but this is a known, measured cost to address before "
              "putting this on a real-time serving path.")


if __name__ == "__main__":
    main()
