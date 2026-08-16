"""Application-level tests for the fraud-prediction views.

Phase 4: Django is a client of the FastAPI scoring service
(users.services.prediction_service.predict_event), so these tests mock that
call rather than hitting a real HTTP service. Covers: the sandbox form's
happy path, a clear degraded message when the risk service is unreachable
(never a 500), per-user data isolation on /database/, and anonymous-user
redirects.
"""
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import Client, TestCase
from django.urls import reverse

from users.models import UserPredictModel
from users.services.prediction_service import RiskAPIUnavailable

VALID_PAYLOAD = {
    'payer_vpa': 'alice@okaxis',
    'payee_vpa': 'merchant@ybl',
    'amount': '500',
    'txn_type': 'P2P',
    'initiation_mode': 'INTENT',
    'device_id': 'device-test123',
    'payer_bank': 'AXIS',
    'payee_bank': 'YBL',
    'payer_account_age_days': '365',
}

FAKE_ALLOW_RESULT = {
    'txn_id': 'fake-txn-1',
    'risk_score': 0.01,
    'action': 'ALLOW',
    'risk_tier': 'LOW',
    'reason_codes': [],
    'model_version': 'warm-fake',
    'thresholds_version': 'fakever1',
    'is_cold': False,
    'latency_ms': 5.0,
    'features_computed': 51,
}


class PredictionFlowTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='alice', password='pw12345!')
        self.client = Client()
        self.client.force_login(self.user)

    def test_prediction_flow_saves_result_for_the_submitting_user(self):
        with patch('users.views.prediction.predict_event', return_value=FAKE_ALLOW_RESULT):
            response = self.client.post(reverse('analyze_upi'), VALID_PAYLOAD)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'ALLOW')
        record = UserPredictModel.objects.get(user=self.user)
        self.assertEqual(record.action, 'ALLOW')
        self.assertEqual(record.risk_score, 0.01)
        self.assertEqual(record.payer_vpa, 'alice@okaxis')

    def test_invalid_amount_does_not_crash_and_shows_message(self):
        response = self.client.post(
            reverse('analyze_upi'), dict(VALID_PAYLOAD, amount='not-a-number'), follow=True,
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Could not analyze')
        self.assertEqual(UserPredictModel.objects.filter(user=self.user).count(), 0)

    def test_risk_service_unavailable_shows_degraded_message_not_a_500(self):
        with patch('users.views.prediction.predict_event', side_effect=RiskAPIUnavailable('down')):
            response = self.client.post(reverse('analyze_upi'), VALID_PAYLOAD, follow=True)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'unavailable')
        self.assertEqual(UserPredictModel.objects.filter(user=self.user).count(), 0)

    def test_anonymous_user_is_redirected_to_login(self):
        anon_client = Client()
        response = anon_client.get(reverse('prediction'))
        self.assertEqual(response.status_code, 302)
        self.assertIn(reverse('login'), response.url)


class UserDataIsolationTests(TestCase):
    """Regression test for the /database/ data-exposure bug found in the audit."""

    def setUp(self):
        self.user_a = User.objects.create_user(username='alice', password='pw12345!')
        self.user_b = User.objects.create_user(username='bob', password='pw12345!')

    def test_user_cannot_see_another_users_predictions(self):
        client_a = Client()
        client_a.force_login(self.user_a)
        with patch('users.views.prediction.predict_event', return_value=FAKE_ALLOW_RESULT):
            client_a.post(reverse('analyze_upi'), dict(VALID_PAYLOAD, amount='42420'))

        client_b = Client()
        client_b.force_login(self.user_b)
        response_b = client_b.get(reverse('database'))

        self.assertEqual(response_b.status_code, 200)
        self.assertEqual(UserPredictModel.objects.filter(user=self.user_b).count(), 0)

    def test_user_sees_only_their_own_predictions(self):
        client_a = Client()
        client_a.force_login(self.user_a)
        with patch('users.views.prediction.predict_event', return_value=FAKE_ALLOW_RESULT):
            client_a.post(reverse('analyze_upi'), dict(VALID_PAYLOAD, amount='42420'))

        response_a = client_a.get(reverse('database'))
        self.assertEqual(UserPredictModel.objects.filter(user=self.user_a).count(), 1)
        self.assertEqual(response_a.status_code, 200)

    def test_anonymous_user_cannot_reach_database_view(self):
        anon_client = Client()
        response = anon_client.get(reverse('database'))
        self.assertEqual(response.status_code, 302)
        self.assertIn(reverse('login'), response.url)
