"""Application-level tests for the fraud-prediction views.

Covers the two behaviors fixed in this pass: per-user data isolation on
/database/ (previously a data leak) and safe handling of invalid prediction
input (previously an uncaught 500).
"""
from django.contrib.auth.models import User
from django.test import Client, TestCase
from django.urls import reverse

from users.models import UserPredictModel

VALID_PAYLOAD = {
    'AverageAmountTransactionDay': '100',
    'TransactionAmount': '500',
    'Is_declined': 'N',
    'TotalNumberOfDeclinesDay': '0',
    'isForeignTransaction': 'N',
    'isHighRiskCountry': 'N',
    'DailyChargebackAvgAmt': '0',
    'Six_MonthAvgChbkAmt': '0',
    'Six_MonthChbkFreq': '0',
}


class PredictionFlowTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='alice', password='pw12345!')
        self.client = Client()
        self.client.force_login(self.user)

    def test_prediction_flow_saves_result_for_the_submitting_user(self):
        response = self.client.post(reverse('analyze_upi'), VALID_PAYLOAD)

        self.assertEqual(response.status_code, 200)
        record = UserPredictModel.objects.get(user=self.user)
        self.assertIn(record.isFradulent, {'Y', 'N'})
        self.assertIsNotNone(record.fraud_probability)

    def test_invalid_input_does_not_crash_and_shows_message(self):
        response = self.client.post(reverse('analyze_upi'), {'TransactionAmount': 'not-a-number'}, follow=True)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Could not analyze')
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
        client_a.post(reverse('analyze_upi'), dict(VALID_PAYLOAD, TransactionAmount='4242'))

        client_b = Client()
        client_b.force_login(self.user_b)
        response_b = client_b.get(reverse('database'))

        self.assertEqual(response_b.status_code, 200)
        self.assertNotContains(response_b, '4242')
        self.assertEqual(UserPredictModel.objects.filter(user=self.user_b).count(), 0)

    def test_user_sees_only_their_own_predictions(self):
        client_a = Client()
        client_a.force_login(self.user_a)
        client_a.post(reverse('analyze_upi'), dict(VALID_PAYLOAD, TransactionAmount='4242'))

        response_a = client_a.get(reverse('database'))
        self.assertContains(response_a, '4242')

    def test_anonymous_user_cannot_reach_database_view(self):
        anon_client = Client()
        response = anon_client.get(reverse('database'))
        self.assertEqual(response.status_code, 302)
        self.assertIn(reverse('login'), response.url)
