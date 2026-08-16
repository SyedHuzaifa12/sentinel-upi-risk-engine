"""Evaluation metrics for the fraud classifier.

Fraud detection is an imbalanced binary classification problem, so accuracy
alone is not reported as a sufficient metric — precision/recall/F1/ROC-AUC/
PR-AUC and the confusion matrix are computed together.

compute_metrics() below is the legacy metric function used by the
RandomForest/UPI_FRAUD.csv path ('Y'/'N' string labels) and is untouched.
Everything below the `# --- Phase 2 ---` marker is new: 0/1-labelled
metrics for the cold/warm LightGBM models (ml/src/train.py), extending this
module rather than replacing it, per CLAUDE.md's "Metrics: PR-AUC,
precision@alert-rate, amount-weighted recall. NEVER accuracy" and Phase 2's
own "REMOVE accuracy from all reporting" for the new path specifically.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

POSITIVE_LABEL = 'Y'  # 'fraudulent'


def compute_metrics(y_true, y_pred, y_proba_positive):
    """Compute the standard classification metrics for the positive ('Y') class.

    y_proba_positive: predicted probability of the positive class, used for
    ROC-AUC and PR-AUC (average precision).
    """
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=['N', 'Y']
    ).ravel()

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        'recall': recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        'roc_auc': roc_auc_score((y_true == POSITIVE_LABEL).astype(int), y_proba_positive),
        'average_precision': average_precision_score((y_true == POSITIVE_LABEL).astype(int), y_proba_positive),
        'confusion_matrix': {
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp),
        },
    }


# --- Phase 2 --------------------------------------------------------------
# 0/1-labelled metrics for the cold/warm LightGBM models. y_true is 0/1
# (fraud/legit), y_proba is the predicted probability of fraud.

def pr_auc(y_true, y_proba):
    """PR-AUC (average precision) -- the PRIMARY metric at this project's
    ~0.7% base rate. ROC-AUC is reported only as a secondary metric
    (CLAUDE.md: never accuracy; PR-AUC first)."""
    return float(average_precision_score(y_true, y_proba))


def precision_at_alert_rate(y_true, y_proba, rate):
    """Precision among the top `rate` fraction of scores (e.g. rate=0.01
    for a 1% alert rate)."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n_alerts = max(1, int(round(len(y_proba) * rate)))
    alert_idx = np.argsort(-y_proba)[:n_alerts]
    return float(y_true[alert_idx].mean())


def recall_at_fpr(y_true, y_proba, target_fpr=0.01):
    """Recall at the operating point achieving the largest TPR among
    thresholds whose FPR does not exceed target_fpr."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    valid = fpr <= target_fpr
    if not valid.any():
        return 0.0
    return float(tpr[valid].max())


def amount_weighted_recall(y_true, y_proba, amounts, rate):
    """Fraction of fraudulent VALUE (sum of amount where y_true==1) caught
    among the top `rate` fraction of alerts by score. This is the business
    metric -- print it most prominently."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    amounts = np.asarray(amounts, dtype=float)

    total_fraud_value = amounts[y_true == 1].sum()
    if total_fraud_value == 0:
        return float('nan')

    n_alerts = max(1, int(round(len(y_proba) * rate)))
    alert_idx = np.argsort(-y_proba)[:n_alerts]
    caught_mask = np.zeros(len(y_proba), dtype=bool)
    caught_mask[alert_idx] = True

    caught_value = amounts[(y_true == 1) & caught_mask].sum()
    return float(caught_value / total_fraud_value)


def per_typology_recall(label_typology, y_true, y_proba, rates=(0.01, 0.02, 0.05)):
    """Recall broken out by fraud typology (MULE_FANIN/SCAM_COLLECT/
    ATO_BURST/QR_SWAP), at each of `rates` alert rates -- reported at all
    three rather than one arbitrary operating point, so QR_SWAP's expected
    weakness is visible across the alert-budget range, never hidden."""
    label_typology = np.asarray(label_typology)
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    result = {}
    for rate in rates:
        n_alerts = max(1, int(round(len(y_proba) * rate)))
        alert_idx = np.argsort(-y_proba)[:n_alerts]
        caught_mask = np.zeros(len(y_proba), dtype=bool)
        caught_mask[alert_idx] = True

        rate_key = f"{rate * 100:g}%"
        rate_result = {}
        for typology in sorted(set(label_typology[y_true == 1])):
            typ_mask = (label_typology == typology) & (y_true == 1)
            n_typ = int(typ_mask.sum())
            if n_typ == 0:
                continue
            rate_result[typology] = {
                'recall': float((typ_mask & caught_mask).sum() / n_typ),
                'n': n_typ,
            }
        result[rate_key] = rate_result
    return result


def brier_score(y_true, y_proba):
    return float(brier_score_loss(y_true, y_proba))


def expected_calibration_error(y_true, y_proba, n_bins=10):
    """Standard ECE: sum over bins of (n_bin/N) * |accuracy_bin - confidence_bin|."""
    y_true = np.asarray(y_true, dtype=float)
    y_proba = np.asarray(y_proba, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_proba, bin_edges[1:-1], right=True), 0, n_bins - 1)

    n = len(y_proba)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        confidence = y_proba[mask].mean()
        accuracy = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(accuracy - confidence)
    return float(ece)


def reliability_curve(y_true, y_proba, n_bins=10):
    """Predicted-vs-observed points for a reliability diagram."""
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy='uniform')
    return {'predicted': prob_pred.tolist(), 'observed': prob_true.tolist()}


def bootstrap_ci(y_true, y_proba, metric_fn, extra_arrays=None, n_resamples=1000, seed=42, **metric_kwargs):
    """Bootstrap 95% CI for metric_fn(y_true, y_proba, **extra_arrays, **metric_kwargs),
    resampling rows with replacement (numpy.random.default_rng(seed)).
    `extra_arrays` (e.g. {"amounts": amounts}) are resampled by the same
    row indices as y_true/y_proba each iteration -- required for a metric
    like amount_weighted_recall, whose extra array must stay row-aligned
    with the resampled labels/scores, not held fixed at its original order.
    Returns (point_estimate, ci_low, ci_high) from the 2.5th/97.5th
    percentiles of the resampled distribution.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    extra_arrays = {k: np.asarray(v) for k, v in (extra_arrays or {}).items()}
    n = len(y_true)

    point_estimate = metric_fn(y_true, y_proba, **extra_arrays, **metric_kwargs)

    rng = np.random.default_rng(seed)
    resampled = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resampled_extra = {k: v[idx] for k, v in extra_arrays.items()}
        resampled[i] = metric_fn(y_true[idx], y_proba[idx], **resampled_extra, **metric_kwargs)

    ci_low, ci_high = np.percentile(resampled, [2.5, 97.5])
    return float(point_estimate), float(ci_low), float(ci_high)


def format_ci(point_estimate, ci_low, ci_high):
    return f"{point_estimate:.2f} [{ci_low:.2f}-{ci_high:.2f}]"


def rules_baseline_score(df):
    """A transparent, no-ML risk score -- an explicit comparison row against
    the trained models, to show ML lift over rules directly. Higher = more
    suspicious. Not a calibrated probability, which is fine: PR-AUC,
    precision@k, and amount-weighted recall are all rank-based and only
    need a monotonic score, not a true probability.
    """
    zscore = df['amount_zscore_vs_payer_30d'].fillna(0)
    return (
        2 * df['payee_is_unseen'].astype(int)
        + 2 * (zscore > 3).astype(int)
        + 1 * df['is_collect_request'].astype(int)
        + 1 * df['is_night'].astype(int)
        + 1 * (df['payer_payee_txn_count'] == 0).astype(int)
    )
