"""The review queue's core invariant: writing an analyst label NEVER mutates
the source `decisions` row -- only creates a new `ReviewLabel` row. This is
what "append-only decisions" (decisionlog/schema.py's DB-level triggers) is
FOR: the review workflow must respect that guarantee at the application
layer too, not just rely on the trigger to reject an accidental UPDATE.

`Decision` is Meta.managed=False (models.py) -- its table isn't part of
Django's migrations, so this test creates it directly via the same
CREATE TABLE IF NOT EXISTS decisionlog/schema.py itself runs, scoped to
Django's test database.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from django.contrib.auth.models import User
from django.db import connection
from django.test import Client, TestCase
from django.urls import reverse

from decisionlog.schema import _CREATE_TABLE_SQL  # noqa: E402
from users.models import Decision, ReviewLabel


class ReviewLabelDoesNotMutateDecisionTests(TestCase):
    def setUp(self):
        with connection.cursor() as cursor:
            cursor.execute(_CREATE_TABLE_SQL)

        self.decision = Decision.objects.create(
            txn_id="txn-review-test-1",
            event={"payer_vpa": "alice@okaxis", "payee_vpa": "shop@ybl", "amount": 500.0},
            feature_snapshot={"amount_log": 6.2, "hour_of_day": 10},
            risk_score=0.82,
            raw_score=0.79,
            is_cold=False,
            action="REVIEW",
            risk_tier="HIGH",
            reason_codes=[{"message": "unusually large amount for this payer"}],
            model_version="warm-test",
            thresholds_version="testver1",
            feature_lib_version="test-sha",
            latency_ms=12.5,
            scored_at=datetime.now(timezone.utc),
            source="worker",
        )

        self.analyst = User.objects.create_user(username="analyst1", password="pw12345!")
        self.client = Client()
        self.client.force_login(self.analyst)

    def _decision_snapshot(self):
        d = Decision.objects.get(pk=self.decision.pk)
        return {
            "txn_id": d.txn_id, "event": d.event, "feature_snapshot": d.feature_snapshot,
            "risk_score": d.risk_score, "raw_score": d.raw_score, "is_cold": d.is_cold,
            "action": d.action, "risk_tier": d.risk_tier, "reason_codes": d.reason_codes,
            "model_version": d.model_version, "thresholds_version": d.thresholds_version,
            "feature_lib_version": d.feature_lib_version, "latency_ms": d.latency_ms,
            "scored_at": d.scored_at, "source": d.source,
        }

    def test_label_write_leaves_decision_row_unchanged(self):
        before = self._decision_snapshot()

        response = self.client.post(reverse("review_queue"), {
            "decision_id": self.decision.pk,
            "disposition": "CONFIRMED_FRAUD",
            "notes": "confirmed with payer over phone",
        })

        self.assertEqual(response.status_code, 302)
        self.assertEqual(ReviewLabel.objects.count(), 1)
        label = ReviewLabel.objects.get()
        self.assertEqual(label.decision_id, self.decision.pk)
        self.assertEqual(label.disposition, "CONFIRMED_FRAUD")
        self.assertEqual(label.reviewer, self.analyst)

        after = self._decision_snapshot()
        self.assertEqual(before, after)

    def test_labeled_decision_drops_out_of_the_pending_queue(self):
        self.client.post(reverse("review_queue"), {
            "decision_id": self.decision.pk,
            "disposition": "LEGIT",
            "notes": "",
        })

        response = self.client.get(reverse("review_queue"))
        self.assertEqual(response.context["queue_depth"], 0)
