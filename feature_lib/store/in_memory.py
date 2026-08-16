"""In-process HistoryStore, designed to be fast at backfill scale (hundreds
of thousands of events), not just correct.

Storage is per-entity (one history per payer_vpa / payee_vpa / device_id /
(payer_vpa, payee_vpa) pair), never a single global list -- a query for
payer X only ever touches payer X's own (typically small) history, never
the full event set. Within an entity's history, `as_of`/window cutoffs are
located with `bisect` in O(log k) (k = that entity's history length), and
sums/means/variances are read off incremental prefix-sum arrays in O(1) via
prefix difference, built up one append at a time in `record()`. Distinct
counts and medians operate on the bounded slice `bisect` found -- bounded by
that one entity's window-local history, never global n.

Precondition: `record()` must be called in non-decreasing event timestamp
order (true both for real-time serving and for `compute_features_batch`,
which processes events in the same order they're computed against). Every
bisect above relies on each entity's own timestamp list already being
sorted; this precondition is what keeps it sorted without ever re-sorting.
"""
import statistics
from bisect import bisect_left
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Optional

from .base import AmountStats, HistoricalTxn, HistoryStore


def _epoch(dt: datetime) -> float:
    return dt.timestamp()


def _from_epoch(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


class _EntityHistory:
    """One entity's (payer/payee/device) transaction history, as parallel
    arrays plus incremental prefix sums for O(1) windowed aggregates."""

    __slots__ = (
        "timestamps", "counterparts", "amounts", "hours",
        "cum_amount", "cum_amount_sq",
        "is_p2m", "cum_p2m",
        "is_collect", "cum_collect",
        "cum_new_counterpart",
        "_seen_counterparts",
    )

    def __init__(self):
        self.timestamps = []
        self.counterparts = []
        self.amounts = []
        self.hours = []
        self.cum_amount = [0.0]
        self.cum_amount_sq = [0.0]
        self.is_p2m = []
        self.cum_p2m = [0]
        self.is_collect = []
        self.cum_collect = [0]
        self.cum_new_counterpart = [0]
        self._seen_counterparts = set()

    def append(self, ts_epoch, counterpart, amount, is_p2m, is_collect, hour):
        is_new = counterpart not in self._seen_counterparts
        self._seen_counterparts.add(counterpart)

        self.timestamps.append(ts_epoch)
        self.counterparts.append(counterpart)
        self.amounts.append(amount)
        self.hours.append(hour)
        self.cum_amount.append(self.cum_amount[-1] + amount)
        self.cum_amount_sq.append(self.cum_amount_sq[-1] + amount * amount)
        self.is_p2m.append(is_p2m)
        self.cum_p2m.append(self.cum_p2m[-1] + (1 if is_p2m else 0))
        self.is_collect.append(is_collect)
        self.cum_collect.append(self.cum_collect[-1] + (1 if is_collect else 0))
        self.cum_new_counterpart.append(self.cum_new_counterpart[-1] + (1 if is_new else 0))

    def bounds(self, as_of_epoch, window_seconds):
        hi = bisect_left(self.timestamps, as_of_epoch)
        lo = 0 if window_seconds is None else bisect_left(self.timestamps, as_of_epoch - window_seconds)
        return lo, hi

    def amount_stats(self, lo, hi):
        count = hi - lo
        if count == 0:
            return AmountStats(count=0, mean=None, std=None, median=None)
        total = self.cum_amount[hi] - self.cum_amount[lo]
        total_sq = self.cum_amount_sq[hi] - self.cum_amount_sq[lo]
        mean = total / count
        variance = max(0.0, total_sq / count - mean * mean)
        return AmountStats(count=count, mean=mean, std=variance ** 0.5, median=statistics.median(self.amounts[lo:hi]))


class _TimestampHistory:
    """A bare sorted-timestamp history for entities that only ever need
    count-before/last-before queries (pairs, and payer-device usage)."""

    __slots__ = ("timestamps",)

    def __init__(self):
        self.timestamps = []

    def append(self, ts_epoch):
        self.timestamps.append(ts_epoch)

    def count_before(self, as_of_epoch):
        return bisect_left(self.timestamps, as_of_epoch)

    def last_before(self, as_of_epoch):
        hi = bisect_left(self.timestamps, as_of_epoch)
        return self.timestamps[hi - 1] if hi > 0 else None


def _window_seconds(window: Optional[timedelta]) -> Optional[float]:
    return None if window is None else window.total_seconds()


class InMemoryHistoryStore(HistoryStore):
    def __init__(self):
        self._payer_histories: dict[str, _EntityHistory] = {}
        self._payee_histories: dict[str, _EntityHistory] = {}
        self._device_histories: dict[str, _EntityHistory] = {}
        self._pair_histories: dict[tuple, _TimestampHistory] = {}
        self._payer_device_histories: dict[tuple, _TimestampHistory] = {}

    # -- payer --------------------------------------------------------

    def payer_txn_count(self, payer_vpa, as_of, window):
        hist = self._payer_histories.get(payer_vpa)
        if hist is None:
            return 0
        lo, hi = hist.bounds(_epoch(as_of), _window_seconds(window))
        return hi - lo

    def payer_distinct_payees(self, payer_vpa, as_of, window):
        hist = self._payer_histories.get(payer_vpa)
        if hist is None:
            return 0
        lo, hi = hist.bounds(_epoch(as_of), _window_seconds(window))
        return len(set(hist.counterparts[lo:hi]))

    def payer_last_txn(self, payer_vpa, as_of):
        hist = self._payer_histories.get(payer_vpa)
        if hist is None:
            return None
        hi = bisect_left(hist.timestamps, _epoch(as_of))
        if hi == 0:
            return None
        idx = hi - 1
        return HistoricalTxn(timestamp=_from_epoch(hist.timestamps[idx]), amount=hist.amounts[idx],
                              payee_vpa=hist.counterparts[idx])

    def payer_amount_stats(self, payer_vpa, as_of, window):
        hist = self._payer_histories.get(payer_vpa)
        if hist is None:
            return AmountStats(count=0, mean=None, std=None, median=None)
        lo, hi = hist.bounds(_epoch(as_of), _window_seconds(window))
        return hist.amount_stats(lo, hi)

    def payer_hour_histogram(self, payer_vpa, as_of, window):
        hist = self._payer_histories.get(payer_vpa)
        if hist is None:
            return {}
        lo, hi = hist.bounds(_epoch(as_of), _window_seconds(window))
        return dict(Counter(hist.hours[lo:hi]))

    def payer_new_payee_txn_ratio(self, payer_vpa, as_of, window):
        hist = self._payer_histories.get(payer_vpa)
        if hist is None:
            return (0, 0)
        lo, hi = hist.bounds(_epoch(as_of), _window_seconds(window))
        new_count = hist.cum_new_counterpart[hi] - hist.cum_new_counterpart[lo]
        return (new_count, hi - lo)

    def payer_collect_request_ratio(self, payer_vpa, as_of, window):
        hist = self._payer_histories.get(payer_vpa)
        if hist is None:
            return (0, 0)
        lo, hi = hist.bounds(_epoch(as_of), _window_seconds(window))
        collect_count = hist.cum_collect[hi] - hist.cum_collect[lo]
        return (collect_count, hi - lo)

    # -- payee --------------------------------------------------------

    def payee_txn_count(self, payee_vpa, as_of):
        hist = self._payee_histories.get(payee_vpa)
        if hist is None:
            return 0
        return bisect_left(hist.timestamps, _epoch(as_of))

    def payee_first_seen(self, payee_vpa, as_of):
        hist = self._payee_histories.get(payee_vpa)
        if hist is None:
            return None
        hi = bisect_left(hist.timestamps, _epoch(as_of))
        if hi == 0:
            return None
        return _from_epoch(hist.timestamps[0])

    def payee_distinct_payers(self, payee_vpa, as_of, window):
        hist = self._payee_histories.get(payee_vpa)
        if hist is None:
            return 0
        lo, hi = hist.bounds(_epoch(as_of), _window_seconds(window))
        return len(set(hist.counterparts[lo:hi]))

    def payee_amount_stats(self, payee_vpa, as_of, window=None):
        hist = self._payee_histories.get(payee_vpa)
        if hist is None:
            return AmountStats(count=0, mean=None, std=None, median=None)
        lo, hi = hist.bounds(_epoch(as_of), _window_seconds(window))
        return hist.amount_stats(lo, hi)

    def payee_p2m_ratio(self, payee_vpa, as_of):
        hist = self._payee_histories.get(payee_vpa)
        if hist is None:
            return None
        hi = bisect_left(hist.timestamps, _epoch(as_of))
        if hi == 0:
            return None
        return (hist.cum_p2m[hi], hi)

    # -- pair -----------------------------------------------------------

    def pair_txn_count(self, payer_vpa, payee_vpa, as_of):
        hist = self._pair_histories.get((payer_vpa, payee_vpa))
        if hist is None:
            return 0
        return hist.count_before(_epoch(as_of))

    def pair_last_txn_time(self, payer_vpa, payee_vpa, as_of):
        hist = self._pair_histories.get((payer_vpa, payee_vpa))
        if hist is None:
            return None
        ts = hist.last_before(_epoch(as_of))
        return None if ts is None else _from_epoch(ts)

    # -- device -----------------------------------------------------------

    def device_distinct_payers(self, device_id, as_of, window):
        hist = self._device_histories.get(device_id)
        if hist is None:
            return 0
        lo, hi = hist.bounds(_epoch(as_of), _window_seconds(window))
        return len(set(hist.counterparts[lo:hi]))

    def device_txn_count(self, device_id, as_of, window):
        hist = self._device_histories.get(device_id)
        if hist is None:
            return 0
        lo, hi = hist.bounds(_epoch(as_of), _window_seconds(window))
        return hi - lo

    def device_seen_for_payer(self, payer_vpa, device_id, as_of):
        hist = self._payer_device_histories.get((payer_vpa, device_id))
        if hist is None:
            return False
        return hist.count_before(_epoch(as_of)) > 0

    # -- write ------------------------------------------------------------

    def record(self, event):
        ts_epoch = _epoch(event.timestamp)
        hour = event.timestamp.hour
        is_p2m = event.txn_type == "P2M"
        is_collect = event.initiation_mode == "COLLECT_REQUEST"

        self._payer_histories.setdefault(event.payer_vpa, _EntityHistory()).append(
            ts_epoch, event.payee_vpa, event.amount, is_p2m, is_collect, hour)
        self._payee_histories.setdefault(event.payee_vpa, _EntityHistory()).append(
            ts_epoch, event.payer_vpa, event.amount, is_p2m, is_collect, hour)
        self._device_histories.setdefault(event.device_id, _EntityHistory()).append(
            ts_epoch, event.payer_vpa, event.amount, is_p2m, is_collect, hour)
        self._pair_histories.setdefault((event.payer_vpa, event.payee_vpa), _TimestampHistory()).append(ts_epoch)
        self._payer_device_histories.setdefault(
            (event.payer_vpa, event.device_id), _TimestampHistory()).append(ts_epoch)
