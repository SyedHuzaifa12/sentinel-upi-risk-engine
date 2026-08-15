"""Group A -- context features. Zero history needed, always available:
computed purely from the event itself, so none of these can be missing."""
import math


def _amount_roundness(amount: float) -> str:
    cents = round(amount * 100)
    if cents % 100 != 0:
        return "neither"
    rupees = cents // 100
    if rupees % 1000 == 0:
        return "thousand"
    if rupees % 100 == 0:
        return "hundred"
    return "neither"


def amount_log(event, store):
    return (math.log1p(event.amount), False)


def hour_of_day(event, store):
    return (event.timestamp.hour, False)


def is_night(event, store):
    hour = event.timestamp.hour
    return (hour >= 23 or hour < 5, False)


def day_of_week(event, store):
    return (event.timestamp.weekday(), False)


def is_collect_request(event, store):
    return (event.initiation_mode == "COLLECT_REQUEST", False)


def is_p2p(event, store):
    return (event.txn_type == "P2P", False)


def amount_roundness(event, store):
    return (_amount_roundness(event.amount), False)


def payer_account_age_days(event, store):
    return (event.payer_account_age_days, False)
