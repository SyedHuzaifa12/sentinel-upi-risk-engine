"""Group D -- payee track record features. Warm only: every one of these
is NaN + _is_missing whenever the payee is literally unseen (count == 0).
All cold_safe=False.

Note this is a *different, finer-grained* gate than compute.is_cold()'s
model-routing threshold (count < COLD_PAYEE_TXN_THRESHOLD, default 3): a
payee with 1-2 prior transactions is routed to the cold model (too little
data to be statistically meaningful), but here that just means these
Group-D values sit unused in the vector rather than driving training --
the cold model never trains on Group D at all. payee_amount_std and
payee_p2m_ratio go further still, going missing under their own count < 2
check, since a single observation can't support a variance/ratio estimate.
"""
from datetime import timedelta

_1H = timedelta(hours=1)
_24H = timedelta(hours=24)


def _payee_is_cold(event, store):
    return store.payee_txn_count(event.payee_vpa, event.timestamp) == 0


def payee_distinct_payers_1h(event, store):
    if _payee_is_cold(event, store):
        return (float("nan"), True)
    return (store.payee_distinct_payers(event.payee_vpa, event.timestamp, _1H), False)


def payee_distinct_payers_24h(event, store):
    if _payee_is_cold(event, store):
        return (float("nan"), True)
    return (store.payee_distinct_payers(event.payee_vpa, event.timestamp, _24H), False)


def payee_distinct_payers_since_first_seen(event, store):
    """Unscaled raw fan-in count, alongside the floored/scaled
    payee_fanin_velocity rate below -- required so a model sees both the
    rate and the raw count it was derived from."""
    first_seen = store.payee_first_seen(event.payee_vpa, event.timestamp)
    if first_seen is None:
        return (float("nan"), True)
    return (store.payee_distinct_payers(event.payee_vpa, event.timestamp, None), False)


def payee_fanin_velocity(event, store):
    first_seen = store.payee_first_seen(event.payee_vpa, event.timestamp)
    if first_seen is None:
        return (float("nan"), True)
    distinct = store.payee_distinct_payers(event.payee_vpa, event.timestamp, None)
    hours_elapsed = (event.timestamp - first_seen).total_seconds() / 3600.0
    # Floored at 1.0h: without this, a payee seen twice 2 minutes apart
    # reports an absurd rate (e.g. 30/hour) purely from a tiny denominator.
    denom = max(1.0, hours_elapsed)
    return (distinct / denom, False)


def payee_amount_mean(event, store):
    stats = store.payee_amount_stats(event.payee_vpa, event.timestamp, None)
    if stats.count == 0:
        return (float("nan"), True)
    return (stats.mean, False)


def payee_amount_std(event, store):
    stats = store.payee_amount_stats(event.payee_vpa, event.timestamp, None)
    if stats.count < 2:
        return (float("nan"), True)
    return (stats.std, False)


def payee_p2m_ratio(event, store):
    result = store.payee_p2m_ratio(event.payee_vpa, event.timestamp)
    if result is None:
        return (float("nan"), True)
    p2m_count, total = result
    if total < 2:
        # A single prior transaction gives a fake-confident 0.0 or 1.0
        # ratio, not a real estimate -- treat it as missing, same as
        # payee_amount_std above.
        return (float("nan"), True)
    return (p2m_count / total, False)


def payer_payee_txn_count(event, store):
    # Spec-mandated: 0 = first payment between this pair, not a missing value.
    return (store.pair_txn_count(event.payer_vpa, event.payee_vpa, event.timestamp), False)


def days_since_payer_last_paid_payee(event, store):
    last = store.pair_last_txn_time(event.payer_vpa, event.payee_vpa, event.timestamp)
    if last is None:
        return (float("nan"), True)
    return ((event.timestamp - last).total_seconds() / 86400.0, False)
