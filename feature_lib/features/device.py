"""Group E -- device features."""
from datetime import timedelta

_1H = timedelta(hours=1)
_30D = timedelta(days=30)


def device_distinct_payers_30d(event, store):
    return (store.device_distinct_payers(event.device_id, event.timestamp, _30D), False)


def device_txn_count_1h(event, store):
    return (store.device_txn_count(event.device_id, event.timestamp, _1H), False)


def is_new_device_for_payer(event, store):
    return (not store.device_seen_for_payer(event.payer_vpa, event.device_id, event.timestamp), False)
