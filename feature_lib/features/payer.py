"""Group B -- payer behaviour features. Need payer history; degrade
gracefully (NaN + _is_missing) when the payer is new or has too little
history for a given computation to be meaningful."""
import math
from datetime import timedelta

_1H = timedelta(hours=1)
_24H = timedelta(hours=24)
_30D = timedelta(days=30)


def amount_zscore_vs_payer_30d(event, store):
    stats = store.payer_amount_stats(event.payer_vpa, event.timestamp, _30D)
    if stats.count < 2 or not stats.std:
        return (float("nan"), True)
    return ((event.amount - stats.mean) / stats.std, False)


def amount_ratio_to_payer_median(event, store):
    stats = store.payer_amount_stats(event.payer_vpa, event.timestamp, _30D)
    if stats.count == 0 or not stats.median:
        return (float("nan"), True)
    return (event.amount / stats.median, False)


def payer_txn_count_1h(event, store):
    return (store.payer_txn_count(event.payer_vpa, event.timestamp, _1H), False)


def payer_txn_count_24h(event, store):
    return (store.payer_txn_count(event.payer_vpa, event.timestamp, _24H), False)


def payer_txn_count_30d(event, store):
    return (store.payer_txn_count(event.payer_vpa, event.timestamp, _30D), False)


def payer_distinct_payees_24h(event, store):
    return (store.payer_distinct_payees(event.payer_vpa, event.timestamp, _24H), False)


def seconds_since_payer_last_txn(event, store):
    last = store.payer_last_txn(event.payer_vpa, event.timestamp)
    if last is None:
        return (float("nan"), True)
    return ((event.timestamp - last.timestamp).total_seconds(), False)


def hour_deviation_from_payer_normal(event, store):
    histogram = store.payer_hour_histogram(event.payer_vpa, event.timestamp, _30D)
    if not histogram:
        return (float("nan"), True)

    total = sum(histogram.values())
    sin_sum = sum(count * math.sin(2 * math.pi * hour / 24) for hour, count in histogram.items())
    cos_sum = sum(count * math.cos(2 * math.pi * hour / 24) for hour, count in histogram.items())
    mean_angle = math.atan2(sin_sum / total, cos_sum / total)
    normal_hour = (mean_angle / (2 * math.pi)) * 24 % 24

    diff = abs(event.timestamp.hour - normal_hour)
    circular_diff = min(diff, 24 - diff)
    return (circular_diff, False)


def payer_new_payee_rate_30d(event, store):
    new_count, total = store.payer_new_payee_txn_ratio(event.payer_vpa, event.timestamp, _30D)
    if total == 0:
        return (float("nan"), True)
    return (new_count / total, False)


def payer_collect_request_rate_30d(event, store):
    collect_count, total = store.payer_collect_request_ratio(event.payer_vpa, event.timestamp, _30D)
    if total == 0:
        return (float("nan"), True)
    return (collect_count / total, False)


def is_first_ever_txn_by_payer(event, store):
    return (store.payer_txn_count(event.payer_vpa, event.timestamp, None) == 0, False)
