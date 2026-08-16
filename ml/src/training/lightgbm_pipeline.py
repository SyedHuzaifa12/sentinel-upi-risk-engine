"""LightGBM model builder + isotonic calibration for the cold/warm models.

Regularized deliberately (not LightGBM's defaults): the cold model trains on
only ~4k rows / ~227 positives (see PROGRESS.md's threshold=3 analysis), and
unregularized boosting would overfit that badly.
"""
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.isotonic import IsotonicRegression

RANDOM_STATE = 42
MIN_TRAIN_POSITIVES = 50

# Raised from 0.40 to 0.50 on 2026-08-16, after the days_since_payer_last_paid_payee
# dominance finding (65.5% -> 42.9% post Phase-0-fix) was validated by ablation:
# removing days_since_payer_last_paid_payee (+ its _is_missing indicator) and
# payer_payee_txn_count from the warm model retained a substantial share of the
# full model's PR-AUC (see metrics_v2.json "ablation_no_pair_history" and
# PROGRESS.md "Decisions (do not revisit)") -- proving the model uses diverse
# signal, not a single-feature crutch. The artificial mechanism that originally
# produced 65.5% (all 4 typologies sharing one "payee new to payer" tell) was
# fixed in the generator; payee-relationship history legitimately being a strong
# fraud signal is expected, not itself evidence of a leak. The guard stays
# active at the new threshold -- this is a considered ceiling, not a removal.
FEATURE_IMPORTANCE_DOMINANCE_THRESHOLD = 0.50

# n_jobs=1 + deterministic=True + force_row_wise=True: LightGBM's default
# multi-threaded histogram building is NOT bit-for-bit reproducible run to
# run otherwise -- required for "same seed -> identical metrics."
LGBM_PARAMS = dict(
    objective='binary',
    num_leaves=31,
    max_depth=6,
    min_child_samples=50,
    learning_rate=0.05,
    n_estimators=2000,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=1,
    deterministic=True,
    force_row_wise=True,
)


def build_model(scale_pos_weight):
    return LGBMClassifier(scale_pos_weight=scale_pos_weight, **LGBM_PARAMS)


def fit_model(model, X_train, y_train, X_val, y_val, categorical_feature=None,
              early_stopping_rounds=100):
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='average_precision',
        categorical_feature=categorical_feature or 'auto',
        callbacks=[
            early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            log_evaluation(period=0),
        ],
    )
    return model


def fit_calibrator(y_val_true, y_val_proba_raw):
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(y_val_proba_raw, y_val_true)
    return calibrator


def apply_calibrator(calibrator, y_proba_raw):
    return calibrator.predict(y_proba_raw)


def check_positive_counts(cold_train_positives, warm_train_positives, min_positives=MIN_TRAIN_POSITIVES):
    """CLAUDE.md/Phase 2: 'If either subset has fewer than 50 positive
    examples, stop and report it rather than training a meaningless model.'
    Checks both before fitting anything -- if either fails, neither model
    is trained."""
    if cold_train_positives < min_positives or warm_train_positives < min_positives:
        raise ValueError(
            f"Refusing to train: cold_train_positives={cold_train_positives}, "
            f"warm_train_positives={warm_train_positives}, minimum required={min_positives}. "
            "At least one subset has too few positive examples for a meaningful model."
        )


def check_feature_importance_dominance(model, feature_names, model_name,
                                        threshold=FEATURE_IMPORTANCE_DOMINANCE_THRESHOLD):
    """Top-15 gain importance, printed always. Raises if any single feature
    exceeds `threshold` share of total gain -- a possible leak, needing
    review before any artifact is saved. amount/amount_log ranking high is
    expected (Phase 0's single-column AUC was ~0.73) and not automatically a
    leak; this check is about *dominance*, not about a strong feature
    ranking first."""
    importances = model.booster_.feature_importance(importance_type='gain')
    total = importances.sum()
    pairs = sorted(zip(feature_names, importances), key=lambda p: p[1], reverse=True)

    print(f"\nTop-15 gain importance ({model_name} model):")
    for name, gain in pairs[:15]:
        share = gain / total if total else 0.0
        print(f"  {name:<45} {share:.1%}")

    if total <= 0:
        return
    top_name, top_gain = pairs[0]
    top_share = top_gain / total
    if top_share > threshold:
        raise ValueError(
            f"Feature '{top_name}' accounts for {top_share:.1%} of total gain in the "
            f"{model_name} model, exceeding the {threshold:.0%} dominance threshold. "
            "Stopping for review before saving any artifact -- this MAY be a leak, or may "
            "be a legitimate strong feature (e.g. amount/amount_log); either way it needs "
            "a human look, not a silent save."
        )
