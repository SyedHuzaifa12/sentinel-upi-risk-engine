"""Evaluation metrics for the fraud classifier.

Fraud detection is an imbalanced binary classification problem, so accuracy
alone is not reported as a sufficient metric — precision/recall/F1/ROC-AUC/
PR-AUC and the confusion matrix are computed together.
"""
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
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
