"""Cost matrix for the decision policy.

These are illustrative estimates for building and exercising a cost-based
threshold optimizer -- NOT measured values from any real operation. A real
deployment would replace every constant here with numbers from actual
support-ticket costs, chargeback data, and analyst time tracking.
"""


def FN_COST(amount: float) -> float:
    """A missed fraud costs the transaction's own value -- the direct loss
    to whoever bears it (payer/bank), with no additional multiplier. This
    is the one cost in this matrix that isn't a guess: it's definitional."""
    return amount


# INR. Illustrative estimate of one analyst's time to manually review a
# flagged transaction (look up the account, call if needed, decide) -- not
# measured from any real analyst-review time-tracking system.
FP_COST_REVIEW = 150.0

# INR. Illustrative estimate of the expected cost of one false WARN shown
# to a legitimate payer -- a small chance they contact support, plus minor
# UX friction/trust erosion. Not measured.
FP_COST_FRICTION = 40.0

# INR. Illustrative estimate of the cost of blocking a genuine payment
# outright -- support contact cost plus a churn-risk proxy (a blocked
# legitimate payment is a much worse experience than a warning). Not
# measured.
FP_COST_BLOCK = 500.0


def TP_BENEFIT(amount: float) -> float:
    """The prevented loss from correctly stopping fraud -- the flip side of
    FN_COST. Also definitional, not a guess."""
    return amount
