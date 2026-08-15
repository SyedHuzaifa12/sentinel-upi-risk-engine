"""Tests for the evaluation pipeline (ml/src/evaluation/metrics.py)."""
import numpy as np
from django.test import SimpleTestCase

from ml.src.evaluation.metrics import compute_metrics

REQUIRED_KEYS = {'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'average_precision', 'confusion_matrix'}


class ComputeMetricsTests(SimpleTestCase):
    def test_metrics_contains_all_required_keys(self):
        y_true = np.array(['N', 'N', 'Y', 'Y', 'N', 'Y'])
        y_pred = np.array(['N', 'N', 'Y', 'N', 'N', 'Y'])
        y_proba = np.array([0.1, 0.2, 0.9, 0.4, 0.05, 0.8])

        metrics = compute_metrics(y_true, y_pred, y_proba)

        self.assertEqual(REQUIRED_KEYS, set(metrics.keys()))
        for key in ('accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'average_precision'):
            self.assertGreaterEqual(metrics[key], 0.0)
            self.assertLessEqual(metrics[key], 1.0)

    def test_confusion_matrix_totals_match_sample_count(self):
        y_true = np.array(['N', 'N', 'Y', 'Y'])
        y_pred = np.array(['N', 'Y', 'Y', 'N'])
        y_proba = np.array([0.1, 0.6, 0.9, 0.4])

        metrics = compute_metrics(y_true, y_pred, y_proba)
        cm = metrics['confusion_matrix']
        total = cm['true_negative'] + cm['false_positive'] + cm['false_negative'] + cm['true_positive']
        self.assertEqual(total, len(y_true))
