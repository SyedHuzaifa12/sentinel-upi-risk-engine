"""Tests for the ML inference layer (ml/src/inference/predictor.py).

No database access needed here, so these run against SimpleTestCase.
"""
from django.test import SimpleTestCase

from ml.src.inference.predictor import FraudPredictor, InvalidInputError, validate_input

VALID_INPUT = {
    'AverageAmountTransactionDay': 100.0,
    'TransactionAmount': 500.0,
    'Is_declined': 'N',
    'TotalNumberOfDeclinesDay': 0,
    'isForeignTransaction': 'N',
    'isHighRiskCountry': 'N',
    'DailyChargebackAvgAmt': 0.0,
    'Six_MonthAvgChbkAmt': 0.0,
    'Six_MonthChbkFreq': 0,
}


class ModelArtifactTests(SimpleTestCase):
    def test_model_artifact_loads_successfully(self):
        predictor = FraudPredictor()
        self.assertIsNotNone(predictor.pipeline)


class DeterministicPredictionTests(SimpleTestCase):
    def test_fixed_input_produces_deterministic_valid_prediction(self):
        predictor = FraudPredictor()
        result_1 = predictor.predict(VALID_INPUT)
        result_2 = predictor.predict(VALID_INPUT)

        self.assertIn(result_1.label, {'Y', 'N'})
        self.assertEqual(result_1.label, result_2.label)
        self.assertAlmostEqual(result_1.fraud_probability, result_2.fraud_probability)
        self.assertGreaterEqual(result_1.fraud_probability, 0.0)
        self.assertLessEqual(result_1.fraud_probability, 1.0)

    def test_predict_returns_top_features(self):
        predictor = FraudPredictor()
        result = predictor.predict(VALID_INPUT)
        self.assertLessEqual(len(result.top_features), 3)
        for entry in result.top_features:
            self.assertIn('feature', entry)
            self.assertIn('importance', entry)


class InputValidationTests(SimpleTestCase):
    def test_missing_fields_are_rejected(self):
        with self.assertRaises(InvalidInputError):
            validate_input({'AverageAmountTransactionDay': 100.0})

    def test_non_numeric_value_is_rejected(self):
        bad_input = dict(VALID_INPUT, TransactionAmount='not-a-number')
        with self.assertRaises(InvalidInputError):
            validate_input(bad_input)

    def test_invalid_categorical_value_is_rejected(self):
        bad_input = dict(VALID_INPUT, Is_declined='MAYBE')
        with self.assertRaises(InvalidInputError):
            validate_input(bad_input)

    def test_predictor_raises_on_invalid_input_rather_than_crashing(self):
        predictor = FraudPredictor()
        with self.assertRaises(InvalidInputError):
            predictor.predict({'TransactionAmount': 'garbage'})
