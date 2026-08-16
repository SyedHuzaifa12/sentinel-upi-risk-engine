"""Phase 2 training entry point: temporal splits, cold + warm LightGBM
models, isotonic calibration, honest metrics (PR-AUC/precision@alert-rate/
amount-weighted-recall -- never accuracy).

    python ml/src/train.py

Produces:
    ml/artifacts/models/cold_model.pkl, warm_model.pkl
    ml/artifacts/models/cold_calibrator.pkl, warm_calibrator.pkl
    ml/artifacts/models/feature_registry.json
    ml/artifacts/metrics/metrics_v2.json
    ml/artifacts/metrics/reliability_cold.json, reliability_warm.json

Does NOT touch ml/legacy/models/upi_fraud_model.pkl or
ml/artifacts/metrics/metrics.json (the pre-migration v1 RandomForest
artifacts, archived to ml/legacy/ in Phase 4 -- backend/ no longer serves
them; see ml/legacy/README.md).
"""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from feature_lib.compute import compute_features_batch  # noqa: E402
from feature_lib.frame import vector_to_frame  # noqa: E402
from feature_lib.registry import ALL_FEATURES, COLD_FEATURES, REGISTRY  # noqa: E402
from feature_lib.store.in_memory import InMemoryHistoryStore  # noqa: E402
from ml.src.data.load_synthetic import load_synthetic_events  # noqa: E402
from ml.src.evaluation import metrics as ev  # noqa: E402
from ml.src.training.lightgbm_pipeline import (  # noqa: E402
    MIN_TRAIN_POSITIVES,
    apply_calibrator,
    build_model,
    check_feature_importance_dominance,
    check_positive_counts,
    fit_calibrator,
    fit_model,
)
from ml.src.training.temporal_split import (  # noqa: E402
    TEST_DAY_RANGE,
    TRAIN_DAY_RANGE,
    VAL_DAY_RANGE,
    assert_temporal_integrity,
    split_by_day,
)
from ml.src.utils.paths import (  # noqa: E402
    COLD_CALIBRATOR_PATH,
    COLD_MODEL_PATH,
    FEATURE_REGISTRY_PATH,
    METRICS_V2_PATH,
    MODELS_DIR,
    RELIABILITY_COLD_PATH,
    RELIABILITY_WARM_PATH,
    SYNTHETIC_EVENTS_PATH,
    WARM_CALIBRATOR_PATH,
    WARM_MODEL_PATH,
)

GENERATOR_SEED = 42  # the --seed used to produce the committed data/synthetic/events.jsonl
ALERT_RATES = (0.01, 0.02, 0.05)
CATEGORICAL_FEATURES = ['amount_roundness']

# Ablation study (2026-08-16): does the warm model collapse without pair
# history? See PROGRESS.md "Decisions (do not revisit)".
ABLATION_EXCLUDED_FEATURES = [
    'days_since_payer_last_paid_payee',
    'days_since_payer_last_paid_payee_is_missing',
    'payer_payee_txn_count',
]

_COLD_SAFE_BY_NAME = {spec.name: spec.cold_safe for spec in REGISTRY}


def build_feature_dataframe(events):
    """One single compute_features_batch pass over the ENTIRE chronological
    event stream (train+val+test together) -- this is what keeps val
    'warmed' by train and test 'warmed' by train+val without ever manually
    carrying store state across split boundaries, which is exactly where
    most people accidentally leak.

    Feature columns (including the amount_roundness categorical cast) come
    from feature_lib.frame.vector_to_frame -- the same function
    service/scoring.py calls at serving time, so training and serving can
    never silently diverge on column order or dtype handling."""
    store = InMemoryHistoryStore()
    vectors = compute_features_batch(events, store)

    df = vector_to_frame([vector.values for vector in vectors], ALL_FEATURES)
    df['txn_id'] = [event.txn_id for event in events]
    df['timestamp'] = pd.to_datetime([event.timestamp for event in events], utc=True)
    df['amount'] = [event.amount for event in events]
    df['label_is_fraud'] = [int(event.label_is_fraud) for event in events]
    df['label_typology'] = [event.label_typology for event in events]
    df['is_cold'] = [vector.is_cold for vector in vectors]
    return df


def _print_cell_counts(train_df, val_df, test_df):
    print("\nEvent/positive counts per split x cold/warm bucket:")
    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        for bucket_name, bucket_mask in [('cold', split_df['is_cold']), ('warm', ~split_df['is_cold'])]:
            bucket_df = split_df[bucket_mask]
            n = len(bucket_df)
            n_pos = int(bucket_df['label_is_fraud'].sum())
            print(f"  {split_name:<5} x {bucket_name:<4}: n={n:>7}  positives={n_pos:>5}")


def _prepare_X(df, feature_columns):
    return df[feature_columns]


def train_one_model(train_df, val_df, feature_columns, model_name):
    X_train = _prepare_X(train_df, feature_columns)
    y_train = train_df['label_is_fraud'].to_numpy()
    X_val = _prepare_X(val_df, feature_columns)
    y_val = val_df['label_is_fraud'].to_numpy()

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos else 1.0

    model = build_model(scale_pos_weight)
    categorical = [c for c in CATEGORICAL_FEATURES if c in feature_columns]
    fit_model(model, X_train, y_train, X_val, y_val, categorical_feature=categorical)

    print(f"{model_name} model: best_iteration_={model.best_iteration_}, "
          f"scale_pos_weight={scale_pos_weight:.2f}")

    check_feature_importance_dominance(model, feature_columns, model_name)

    val_proba_raw = model.predict_proba(X_val)[:, 1]
    calibrator = fit_calibrator(y_val, val_proba_raw)

    return model, calibrator


def evaluate_model(model, calibrator, test_df, feature_columns, model_name, n_bootstrap=1000):
    X_test = _prepare_X(test_df, feature_columns)
    y_test = test_df['label_is_fraud'].to_numpy()
    amounts = test_df['amount'].to_numpy()
    typologies = test_df['label_typology'].to_numpy()

    proba_raw = model.predict_proba(X_test)[:, 1]
    proba_calibrated = apply_calibrator(calibrator, proba_raw)

    n_pos = int(y_test.sum())
    if model_name == 'cold' and n_pos < 100:
        print(f"\n  WARNING: cold test positives = {n_pos} (<100) -- cold metrics below "
              f"are INDICATIVE, not precise. Treat point estimates with the bootstrap CI in mind.")

    result = {
        'n_test_events': len(test_df),
        'n_test_positives': n_pos,
        'best_iteration': int(model.best_iteration_),
        'brier_raw': ev.brier_score(y_test, proba_raw),
        'brier_calibrated': ev.brier_score(y_test, proba_calibrated),
        'ece_raw': ev.expected_calibration_error(y_test, proba_raw),
        'ece_calibrated': ev.expected_calibration_error(y_test, proba_calibrated),
        'reliability_raw': ev.reliability_curve(y_test, proba_raw),
        'reliability_calibrated': ev.reliability_curve(y_test, proba_calibrated),
        'roc_auc': float(roc_auc_score(y_test, proba_calibrated)),
        'pr_auc': ev.pr_auc(y_test, proba_calibrated),
        'recall_at_1pct_fpr': ev.recall_at_fpr(y_test, proba_calibrated, target_fpr=0.01),
        'precision_at_alert_rate': {},
        'amount_weighted_recall_at_alert_rate': {},
    }

    pr_auc_est, pr_auc_lo, pr_auc_hi = ev.bootstrap_ci(y_test, proba_calibrated, ev.pr_auc, n_resamples=n_bootstrap)
    result['pr_auc_ci'] = ev.format_ci(pr_auc_est, pr_auc_lo, pr_auc_hi)

    for rate in ALERT_RATES:
        rate_key = f"{rate * 100:g}%"
        result['precision_at_alert_rate'][rate_key] = ev.precision_at_alert_rate(y_test, proba_calibrated, rate)
        awr = ev.amount_weighted_recall(y_test, proba_calibrated, amounts, rate)
        result['amount_weighted_recall_at_alert_rate'][rate_key] = awr

        if abs(rate - 0.01) < 1e-9:
            awr_est, awr_lo, awr_hi = ev.bootstrap_ci(
                y_test, proba_calibrated, ev.amount_weighted_recall, n_resamples=n_bootstrap,
                extra_arrays={'amounts': amounts}, rate=rate,
            )
            result['amount_weighted_recall_at_1pct_ci'] = ev.format_ci(awr_est, awr_lo, awr_hi)

    result['per_typology_recall'] = ev.per_typology_recall(typologies, y_test, proba_calibrated, rates=ALERT_RATES)

    return result, proba_calibrated


def evaluate_score_only(df, score, name):
    """For the rules baseline -- no model, no calibration, just the raw
    monotonic score fed through the same rank-based metric functions."""
    y_true = df['label_is_fraud'].to_numpy()
    amounts = df['amount'].to_numpy()
    typologies = df['label_typology'].to_numpy()
    score = np.asarray(score)

    result = {
        'n_test_events': len(df),
        'n_test_positives': int(y_true.sum()),
        'pr_auc': ev.pr_auc(y_true, score),
        'precision_at_alert_rate': {},
        'amount_weighted_recall_at_alert_rate': {},
    }
    for rate in ALERT_RATES:
        rate_key = f"{rate * 100:g}%"
        result['precision_at_alert_rate'][rate_key] = ev.precision_at_alert_rate(y_true, score, rate)
        result['amount_weighted_recall_at_alert_rate'][rate_key] = ev.amount_weighted_recall(
            y_true, score, amounts, rate)
    result['per_typology_recall'] = ev.per_typology_recall(typologies, y_true, score, rates=ALERT_RATES)
    return result


def _git_sha():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[2],
        ).decode().strip()
    except Exception:
        return None


def run_pipeline(df, min_positives=MIN_TRAIN_POSITIVES, n_bootstrap=1000):
    """The full split -> train -> calibrate -> evaluate pipeline. Returns a
    dict with the fitted models/calibrators and every metric, importable so
    tests can call it directly on a hand-built small df."""
    sim_start = df['timestamp'].min()
    train_df, val_df, test_df = split_by_day(df, sim_start)
    assert_temporal_integrity(train_df, val_df, test_df)

    _print_cell_counts(train_df, val_df, test_df)

    n_mule_outbound = int((train_df['label_typology'] == 'MULE_OUTBOUND').sum())
    train_df = train_df[train_df['label_typology'] != 'MULE_OUTBOUND']
    print(f"\nDropped {n_mule_outbound} MULE_OUTBOUND rows from training "
          f"(kept in val/test as ordinary negatives, kept in the event stream for feature computation).")

    cold_train = train_df[train_df['is_cold']]
    warm_train = train_df[~train_df['is_cold']]
    check_positive_counts(
        int(cold_train['label_is_fraud'].sum()), int(warm_train['label_is_fraud'].sum()), min_positives,
    )

    cold_val = val_df[val_df['is_cold']]
    warm_val = val_df[~val_df['is_cold']]
    cold_test = test_df[test_df['is_cold']]
    warm_test = test_df[~test_df['is_cold']]

    cold_model, cold_calibrator = train_one_model(cold_train, cold_val, COLD_FEATURES, 'cold')
    warm_model, warm_calibrator = train_one_model(warm_train, warm_val, ALL_FEATURES, 'warm')

    cold_metrics, cold_proba = evaluate_model(
        cold_model, cold_calibrator, cold_test, COLD_FEATURES, 'cold', n_bootstrap)
    warm_metrics, warm_proba = evaluate_model(
        warm_model, warm_calibrator, warm_test, ALL_FEATURES, 'warm', n_bootstrap)

    # Ablation: does the warm model collapse to a single-feature detector
    # without pair-history? Required before accepting the 42.9% gain
    # dominance of days_since_payer_last_paid_payee as a legitimate strong
    # feature rather than a crutch (see PROGRESS.md "Decisions (do not
    # revisit)"). Also drops the _is_missing companion -- it carries nearly
    # the same information as payer_payee_txn_count==0, so leaving it in
    # would let the model route around the ablation via the missingness
    # flag alone, defeating the point of removing the value column.
    ablation_features = [f for f in ALL_FEATURES if f not in ABLATION_EXCLUDED_FEATURES]
    ablation_model, ablation_calibrator = train_one_model(
        warm_train, warm_val, ablation_features, 'warm_ablation_no_pair_history')
    ablation_metrics, _ = evaluate_model(
        ablation_model, ablation_calibrator, warm_test, ablation_features,
        'warm_ablation_no_pair_history', n_bootstrap)

    pr_auc_retained_share = (
        ablation_metrics['pr_auc'] / warm_metrics['pr_auc'] if warm_metrics['pr_auc'] else float('nan')
    )
    print("\nAblation (no pair history) vs full warm model:")
    print(f"  Excluded features: {ABLATION_EXCLUDED_FEATURES}")
    print(f"  Full warm PR-AUC:     {warm_metrics['pr_auc']:.4f}")
    print(f"  Ablated PR-AUC:       {ablation_metrics['pr_auc']:.4f}  ({pr_auc_retained_share:.1%} retained)")
    print(f"  Full warm precision@1%:   {warm_metrics['precision_at_alert_rate']['1%']:.4f}")
    print(f"  Ablated precision@1%:     {ablation_metrics['precision_at_alert_rate']['1%']:.4f}")
    print(f"  Full warm amt-wtd recall@1%:  {warm_metrics['amount_weighted_recall_at_alert_rate']['1%']:.4f}")
    print(f"  Ablated amt-wtd recall@1%:    {ablation_metrics['amount_weighted_recall_at_alert_rate']['1%']:.4f}")
    ablation_metrics['pr_auc_retained_share_vs_full_warm'] = pr_auc_retained_share
    ablation_metrics['excluded_features'] = ABLATION_EXCLUDED_FEATURES

    combined_y = np.concatenate([cold_test['label_is_fraud'].to_numpy(), warm_test['label_is_fraud'].to_numpy()])
    combined_proba = np.concatenate([cold_proba, warm_proba])
    combined_amounts = np.concatenate([cold_test['amount'].to_numpy(), warm_test['amount'].to_numpy()])
    combined_typologies = np.concatenate(
        [cold_test['label_typology'].to_numpy(), warm_test['label_typology'].to_numpy()])

    combined_metrics = {
        'n_test_events': len(combined_y),
        'n_test_positives': int(combined_y.sum()),
        'pr_auc': ev.pr_auc(combined_y, combined_proba),
        'roc_auc': float(roc_auc_score(combined_y, combined_proba)),
        'recall_at_1pct_fpr': ev.recall_at_fpr(combined_y, combined_proba, target_fpr=0.01),
        'precision_at_alert_rate': {}, 'amount_weighted_recall_at_alert_rate': {},
    }
    for rate in ALERT_RATES:
        rate_key = f"{rate * 100:g}%"
        combined_metrics['precision_at_alert_rate'][rate_key] = ev.precision_at_alert_rate(
            combined_y, combined_proba, rate)
        combined_metrics['amount_weighted_recall_at_alert_rate'][rate_key] = ev.amount_weighted_recall(
            combined_y, combined_proba, combined_amounts, rate)
    combined_metrics['per_typology_recall'] = ev.per_typology_recall(
        combined_typologies, combined_y, combined_proba, rates=ALERT_RATES)

    combined_test_df = pd.concat([cold_test, warm_test])
    rules_score = ev.rules_baseline_score(combined_test_df)
    rules_metrics = evaluate_score_only(combined_test_df, rules_score, 'rules_baseline')

    return {
        'models': {'cold': cold_model, 'warm': warm_model},
        'calibrators': {'cold': cold_calibrator, 'warm': warm_calibrator},
        'metrics': {
            'cold': cold_metrics, 'warm': warm_metrics, 'combined': combined_metrics,
            'rules_baseline': rules_metrics, 'ablation_no_pair_history': ablation_metrics,
        },
        'n_mule_outbound_dropped': n_mule_outbound,
        'sim_start': sim_start,
    }


def _print_comparison_table(metrics_by_row):
    print("\n" + "=" * 100)
    print("FINAL COMPARISON: cold vs warm vs combined vs rules baseline")
    print("=" * 100)
    for row_name, m in metrics_by_row.items():
        print(f"\n-- {row_name} --")
        print(f"  n_test_events={m['n_test_events']}  n_test_positives={m['n_test_positives']}")
        if 'pr_auc_ci' in m:
            print(f"  PR-AUC (primary):        {m['pr_auc_ci']}")
        else:
            print(f"  PR-AUC (primary):        {m['pr_auc']:.4f}")
        if 'roc_auc' in m:
            print(f"  ROC-AUC (secondary):     {m['roc_auc']:.4f}")
        if 'recall_at_1pct_fpr' in m:
            print(f"  Recall@1%% FPR:           {m['recall_at_1pct_fpr']:.4f}")
        for rate_key, val in m['precision_at_alert_rate'].items():
            print(f"  Precision@{rate_key} alert rate: {val:.4f}")
        for rate_key, val in m['amount_weighted_recall_at_alert_rate'].items():
            has_ci = rate_key == '1%' and 'amount_weighted_recall_at_1pct_ci' in m
            suffix = f"  CI={m['amount_weighted_recall_at_1pct_ci']}" if has_ci else ""
            print(f"  Amount-weighted recall@{rate_key}: {val:.4f}{suffix}")
        if 'brier_raw' in m:
            print(f"  Brier raw -> calibrated: {m['brier_raw']:.4f} -> {m['brier_calibrated']:.4f}")
            print(f"  ECE raw -> calibrated:   {m['ece_raw']:.4f} -> {m['ece_calibrated']:.4f}")
        print(f"  Per-typology recall: {json.dumps(m['per_typology_recall'], indent=2)}")


def main():
    print(f"Loading synthetic events from {SYNTHETIC_EVENTS_PATH} ...")
    events = load_synthetic_events()
    print(f"Loaded {len(events)} events.")

    print("Computing features (single pass, train+val+test together) ...")
    df = build_feature_dataframe(events)

    result = run_pipeline(df)

    metrics_by_row = {
        'cold': result['metrics']['cold'],
        'warm': result['metrics']['warm'],
        'combined': result['metrics']['combined'],
        'rules_baseline': result['metrics']['rules_baseline'],
    }
    _print_comparison_table(metrics_by_row)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime('%Y%m%d')
    git_sha = _git_sha()
    sha_short = (git_sha or 'nogit')[:7]

    cold_model = result['models']['cold']
    warm_model = result['models']['warm']
    cold_model.model_version_ = f"cold-{date_str}-{sha_short}"
    warm_model.model_version_ = f"warm-{date_str}-{sha_short}"

    joblib.dump(cold_model, COLD_MODEL_PATH)
    joblib.dump(warm_model, WARM_MODEL_PATH)
    joblib.dump(result['calibrators']['cold'], COLD_CALIBRATOR_PATH)
    joblib.dump(result['calibrators']['warm'], WARM_CALIBRATOR_PATH)

    feature_registry = {
        'cold_model': {
            'features': COLD_FEATURES,
            'cold_safe': [_COLD_SAFE_BY_NAME[name] for name in COLD_FEATURES],
            'model_version': cold_model.model_version_,
        },
        'warm_model': {
            'features': ALL_FEATURES,
            'cold_safe': [_COLD_SAFE_BY_NAME[name] for name in ALL_FEATURES],
            'model_version': warm_model.model_version_,
        },
    }
    with open(FEATURE_REGISTRY_PATH, 'w') as f:
        json.dump(feature_registry, f, indent=2)

    with open(RELIABILITY_COLD_PATH, 'w') as f:
        json.dump({
            'raw': result['metrics']['cold']['reliability_raw'],
            'calibrated': result['metrics']['cold']['reliability_calibrated'],
        }, f, indent=2)
    with open(RELIABILITY_WARM_PATH, 'w') as f:
        json.dump({
            'raw': result['metrics']['warm']['reliability_raw'],
            'calibrated': result['metrics']['warm']['reliability_calibrated'],
        }, f, indent=2)

    metrics_v2 = {
        'trained_at_utc': datetime.now(timezone.utc).isoformat(),
        'generator_seed': GENERATOR_SEED,
        'git_sha': git_sha,
        'library_versions': {
            'lightgbm': lightgbm.__version__,
            'scikit-learn': sklearn.__version__,
            'pandas': pd.__version__,
            'numpy': np.__version__,
        },
        'split_day_ranges': {
            'train': list(TRAIN_DAY_RANGE),
            'val': list(VAL_DAY_RANGE),
            'test': list(TEST_DAY_RANGE),
        },
        'n_mule_outbound_dropped_from_training': result['n_mule_outbound_dropped'],
        'notes': [
            "The validation split is reused for both LightGBM early stopping and isotonic "
            "calibration -- a deliberate, documented choice given dataset size, not an oversight.",
            "Cold-model metrics are based on a small positive sample; see the bootstrap CIs "
            "and the cold-test-positives warning printed above.",
            "FEATURE_IMPORTANCE_DOMINANCE_THRESHOLD raised from 0.40 to 0.50 on 2026-08-16 after "
            "the days_since_payer_last_paid_payee ablation below showed the warm model retains "
            "substantial performance without pair-history features -- see "
            "'ablation_no_pair_history' and PROGRESS.md 'Decisions (do not revisit)'.",
        ],
        'metrics': {
            'cold': {k: v for k, v in result['metrics']['cold'].items()
                     if k not in ('reliability_raw', 'reliability_calibrated')},
            'warm': {k: v for k, v in result['metrics']['warm'].items()
                     if k not in ('reliability_raw', 'reliability_calibrated')},
            'combined': result['metrics']['combined'],
            'rules_baseline': result['metrics']['rules_baseline'],
            'ablation_no_pair_history': {k: v for k, v in result['metrics']['ablation_no_pair_history'].items()
                                         if k not in ('reliability_raw', 'reliability_calibrated')},
        },
    }
    with open(METRICS_V2_PATH, 'w') as f:
        json.dump(metrics_v2, f, indent=2, default=str)

    print(f"\nWrote artifacts to {MODELS_DIR} and {METRICS_V2_PATH.parent}")
    return metrics_v2


if __name__ == '__main__':
    main()
