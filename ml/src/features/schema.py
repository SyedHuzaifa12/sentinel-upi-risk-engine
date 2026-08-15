"""Feature schema for the UPI fraud model.

This is the single source of truth for what the model expects, derived by
introspecting the trained pipeline's ``feature_names_in_`` /
``ColumnTransformer`` configuration. Both training (ml/src/train.py) and
serving (backend/users/services/prediction_service.py, via
ml/src/inference/predictor.py) import from here so the two can never drift
silently.
"""

# Columns dropped before training (present in the raw CSV, not predictive):
# Merchant_id is an identifier; TransactionDate is 100% null in the source data.
DROP_COLUMNS = ['Merchant_id', 'TransactionDate']

NUMERIC_FEATURES = [
    'AverageAmountTransactionDay',
    'TransactionAmount',
    'TotalNumberOfDeclinesDay',
    'DailyChargebackAvgAmt',
    'Six_MonthAvgChbkAmt',
    'Six_MonthChbkFreq',
]

CATEGORICAL_FEATURES = [
    'Is_declined',
    'isForeignTransaction',
    'isHighRiskCountry',
]

# Exact order the trained pipeline expects (matches feature_names_in_).
FEATURE_COLUMNS = [
    'AverageAmountTransactionDay',
    'TransactionAmount',
    'Is_declined',
    'TotalNumberOfDeclinesDay',
    'isForeignTransaction',
    'isHighRiskCountry',
    'DailyChargebackAvgAmt',
    'Six_MonthAvgChbkAmt',
    'Six_MonthChbkFreq',
]

TARGET_COLUMN = 'isFradulent'

# Valid values for every Y/N column in this dataset (features and target).
YES_NO_VALUES = {'Y', 'N'}
