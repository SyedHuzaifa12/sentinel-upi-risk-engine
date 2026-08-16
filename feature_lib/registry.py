"""The feature registry: the single source of truth for every feature this
package computes, and for the cold/warm split.

`BASE_REGISTRY` has one entry per *computed* feature (37 total: the 36 from
the spec, plus payee_distinct_payers_since_first_seen). `REGISTRY` is
mechanically derived from it -- every base entry, plus a second
`<name>_is_missing` entry for every `can_be_missing` base feature -- so
`COLD_FEATURES`/`WARM_ONLY_FEATURES`/`ALL_FEATURES` can never hand-drift
from the per-feature `cold_safe` flags: they're just filters over
`REGISTRY`.

Naming note: there is deliberately no `WARM_FEATURES` name. The WARM model
trains on ALL features (cold-safe ones remain informative once a payee is
established), not just the warm-only ones -- an ambiguous "WARM_FEATURES"
name invited exactly that silent bug, so `WARM_ONLY_FEATURES` is explicit
about being the Group-D-only subset, and `ALL_FEATURES` is what the warm
model actually trains on.
"""
from dataclasses import dataclass
from typing import Callable

from .event import UPIEvent
from .features import context, device, payee_newness, payee_track_record, payer
from .store.base import HistoryStore


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    group: str  # "A".."E"
    cold_safe: bool
    can_be_missing: bool
    compute: Callable[[UPIEvent, HistoryStore], tuple]  # -> (value, is_missing)


BASE_REGISTRY: list[FeatureSpec] = [
    # Group A -- context. Zero history, never missing.
    FeatureSpec("amount_log", "A", True, False, context.amount_log),
    FeatureSpec("hour_of_day", "A", True, False, context.hour_of_day),
    FeatureSpec("is_night", "A", True, False, context.is_night),
    FeatureSpec("day_of_week", "A", True, False, context.day_of_week),
    FeatureSpec("is_collect_request", "A", True, False, context.is_collect_request),
    FeatureSpec("is_p2p", "A", True, False, context.is_p2p),
    FeatureSpec("amount_roundness", "A", True, False, context.amount_roundness),
    FeatureSpec("payer_account_age_days", "A", True, False, context.payer_account_age_days),

    # Group B -- payer behaviour. cold_safe re: payee coldness -- these
    # remain computable/meaningful even when the payee has no history.
    FeatureSpec("amount_zscore_vs_payer_30d", "B", True, True, payer.amount_zscore_vs_payer_30d),
    FeatureSpec("amount_ratio_to_payer_median", "B", True, True, payer.amount_ratio_to_payer_median),
    FeatureSpec("payer_txn_count_1h", "B", True, False, payer.payer_txn_count_1h),
    FeatureSpec("payer_txn_count_24h", "B", True, False, payer.payer_txn_count_24h),
    FeatureSpec("payer_txn_count_30d", "B", True, False, payer.payer_txn_count_30d),
    FeatureSpec("payer_distinct_payees_24h", "B", True, False, payer.payer_distinct_payees_24h),
    FeatureSpec("seconds_since_payer_last_txn", "B", True, True, payer.seconds_since_payer_last_txn),
    FeatureSpec("hour_deviation_from_payer_normal", "B", True, True, payer.hour_deviation_from_payer_normal),
    FeatureSpec("payer_new_payee_rate_30d", "B", True, True, payer.payer_new_payee_rate_30d),
    FeatureSpec("payer_collect_request_rate_30d", "B", True, True, payer.payer_collect_request_rate_30d),
    FeatureSpec("is_first_ever_txn_by_payer", "B", True, False, payer.is_first_ever_txn_by_payer),

    # Group C -- payee newness. The cold-start core.
    FeatureSpec("payee_age_hours_in_system", "C", True, False, payee_newness.payee_age_hours_in_system),
    FeatureSpec("payee_total_txn_count", "C", True, False, payee_newness.payee_total_txn_count),
    FeatureSpec("payee_is_unseen", "C", True, False, payee_newness.payee_is_unseen),
    FeatureSpec("payee_handle_digit_ratio", "C", True, False, payee_newness.payee_handle_digit_ratio),
    FeatureSpec("payee_handle_entropy", "C", True, False, payee_newness.payee_handle_entropy),
    FeatureSpec("payee_handle_has_dictionary_name", "C", True, False, payee_newness.payee_handle_has_dictionary_name),

    # Group D -- payee track record. Warm only.
    FeatureSpec("payee_distinct_payers_1h", "D", False, True, payee_track_record.payee_distinct_payers_1h),
    FeatureSpec("payee_distinct_payers_24h", "D", False, True, payee_track_record.payee_distinct_payers_24h),
    FeatureSpec("payee_distinct_payers_since_first_seen", "D", False, True,
                payee_track_record.payee_distinct_payers_since_first_seen),
    FeatureSpec("payee_fanin_velocity", "D", False, True, payee_track_record.payee_fanin_velocity),
    FeatureSpec("payee_amount_mean", "D", False, True, payee_track_record.payee_amount_mean),
    FeatureSpec("payee_amount_std", "D", False, True, payee_track_record.payee_amount_std),
    FeatureSpec("payee_p2m_ratio", "D", False, True, payee_track_record.payee_p2m_ratio),
    FeatureSpec("payer_payee_txn_count", "D", False, False, payee_track_record.payer_payee_txn_count),
    FeatureSpec("days_since_payer_last_paid_payee", "D", False, True,
                payee_track_record.days_since_payer_last_paid_payee),

    # Group E -- device.
    FeatureSpec("device_distinct_payers_30d", "E", True, False, device.device_distinct_payers_30d),
    FeatureSpec("device_txn_count_1h", "E", True, False, device.device_txn_count_1h),
    FeatureSpec("is_new_device_for_payer", "E", True, False, device.is_new_device_for_payer),
]


def _expand(base_registry: list) -> list:
    expanded = []
    for spec in base_registry:
        expanded.append(spec)
        if spec.can_be_missing:
            expanded.append(FeatureSpec(f"{spec.name}_is_missing", spec.group, spec.cold_safe, False, spec.compute))
    return expanded


REGISTRY: list[FeatureSpec] = _expand(BASE_REGISTRY)

COLD_FEATURES: list[str] = [f.name for f in REGISTRY if f.cold_safe]
WARM_ONLY_FEATURES: list[str] = [f.name for f in REGISTRY if not f.cold_safe]
ALL_FEATURES: list[str] = [f.name for f in REGISTRY]
