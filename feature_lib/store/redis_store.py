"""Redis-backed HistoryStore -- the online/serving implementation.

Each entity (payer/payee/device/pair) gets one Redis sorted set, scored by
event epoch-seconds. PIT correctness lives in the range query: every read
uses an *exclusive* upper bound at `as_of` (Redis range syntax `"(" + value`),
never inclusive. Not imported by `feature_lib.store`'s `__init__.py`, so
importing feature_lib never requires redis-py to be installed -- import
this module directly:

    from feature_lib.store.redis_store import RedisHistoryStore
"""
import statistics
from datetime import timedelta
from typing import Optional

import redis

from feature_lib.event import UPIEvent

from .base import AmountStats, HistoricalTxn, HistoryStore


def _epoch(dt) -> float:
    return dt.timestamp()


def _from_epoch(ts):
    from datetime import datetime, timezone
    return datetime.fromtimestamp(float(ts), tz=timezone.utc)


def _encode(txn_id, counterpart, amount, txn_type, initiation_mode, hour):
    return f"{txn_id}\x1f{counterpart}\x1f{amount}\x1f{txn_type}\x1f{initiation_mode}\x1f{hour}"


def _decode(member: str):
    txn_id, counterpart, amount, txn_type, initiation_mode, hour = member.split("\x1f")
    return {
        "txn_id": txn_id, "counterpart": counterpart, "amount": float(amount),
        "txn_type": txn_type, "initiation_mode": initiation_mode, "hour": int(hour),
    }


def _range_bounds(as_of_epoch, window: Optional[timedelta]):
    hi = f"({as_of_epoch!r}"
    lo = "-inf" if window is None else (as_of_epoch - window.total_seconds())
    return lo, hi


class RedisHistoryStore(HistoryStore):
    def __init__(self, url: str, prefix: str = "feature_lib", flush: bool = False):
        self._client = redis.Redis.from_url(url, decode_responses=True)
        self._client.ping()
        self._prefix = prefix
        if flush:
            for key in self._client.scan_iter(f"{prefix}:*"):
                self._client.delete(key)

    def close(self):
        self._client.close()

    def _payer_key(self, payer_vpa):
        return f"{self._prefix}:payer:{payer_vpa}"

    def _payee_key(self, payee_vpa):
        return f"{self._prefix}:payee:{payee_vpa}"

    def _device_key(self, device_id):
        return f"{self._prefix}:device:{device_id}"

    def _pair_key(self, payer_vpa, payee_vpa):
        return f"{self._prefix}:pair:{payer_vpa}:{payee_vpa}"

    def _payer_device_key(self, payer_vpa, device_id):
        return f"{self._prefix}:payerdevice:{payer_vpa}:{device_id}"

    def _members(self, key, as_of, window):
        lo, hi = _range_bounds(_epoch(as_of), window)
        return [_decode(m) for m in self._client.zrangebyscore(key, lo, hi)]

    # -- payer --------------------------------------------------------

    def payer_txn_count(self, payer_vpa, as_of, window):
        lo, hi = _range_bounds(_epoch(as_of), window)
        return self._client.zcount(self._payer_key(payer_vpa), lo, hi)

    def payer_distinct_payees(self, payer_vpa, as_of, window):
        rows = self._members(self._payer_key(payer_vpa), as_of, window)
        return len({r["counterpart"] for r in rows})

    def payer_last_txn(self, payer_vpa, as_of):
        hi = f"({_epoch(as_of)!r}"
        entries = self._client.zrevrangebyscore(self._payer_key(payer_vpa), hi, "-inf", start=0, num=1,
                                                  withscores=True)
        if not entries:
            return None
        member, score = entries[0]
        row = _decode(member)
        return HistoricalTxn(timestamp=_from_epoch(score), amount=row["amount"], payee_vpa=row["counterpart"])

    def payer_amount_stats(self, payer_vpa, as_of, window):
        rows = self._members(self._payer_key(payer_vpa), as_of, window)
        if not rows:
            return AmountStats(count=0, mean=None, std=None, median=None)
        amounts = [r["amount"] for r in rows]
        return AmountStats(count=len(amounts), mean=statistics.fmean(amounts),
                            std=statistics.pstdev(amounts), median=statistics.median(amounts))

    def payer_hour_histogram(self, payer_vpa, as_of, window):
        rows = self._members(self._payer_key(payer_vpa), as_of, window)
        histogram = {}
        for r in rows:
            histogram[r["hour"]] = histogram.get(r["hour"], 0) + 1
        return histogram

    def payer_new_payee_txn_ratio(self, payer_vpa, as_of, window):
        rows = self._members(self._payer_key(payer_vpa), as_of, window)
        if not rows:
            return (0, 0)
        # "new" relative to this payer's whole history strictly before as_of,
        # not just within the window -- mirrors the in-memory definition.
        all_rows = self._members(self._payer_key(payer_vpa), as_of, None)
        seen = set()
        new_count = 0
        window_start_idx = len(all_rows) - len(rows)
        for i, r in enumerate(all_rows):
            is_new = r["counterpart"] not in seen
            seen.add(r["counterpart"])
            if i >= window_start_idx and is_new:
                new_count += 1
        return (new_count, len(rows))

    def payer_collect_request_ratio(self, payer_vpa, as_of, window):
        rows = self._members(self._payer_key(payer_vpa), as_of, window)
        collect_count = sum(1 for r in rows if r["initiation_mode"] == "COLLECT_REQUEST")
        return (collect_count, len(rows))

    # -- payee --------------------------------------------------------

    def payee_txn_count(self, payee_vpa, as_of):
        lo, hi = _range_bounds(_epoch(as_of), None)
        return self._client.zcount(self._payee_key(payee_vpa), lo, hi)

    def payee_first_seen(self, payee_vpa, as_of):
        hi = f"({_epoch(as_of)!r}"
        entries = self._client.zrangebyscore(self._payee_key(payee_vpa), "-inf", hi, start=0, num=1,
                                              withscores=True)
        if not entries:
            return None
        return _from_epoch(entries[0][1])

    def payee_distinct_payers(self, payee_vpa, as_of, window):
        rows = self._members(self._payee_key(payee_vpa), as_of, window)
        return len({r["counterpart"] for r in rows})

    def payee_amount_stats(self, payee_vpa, as_of, window=None):
        rows = self._members(self._payee_key(payee_vpa), as_of, window)
        if not rows:
            return AmountStats(count=0, mean=None, std=None, median=None)
        amounts = [r["amount"] for r in rows]
        return AmountStats(count=len(amounts), mean=statistics.fmean(amounts),
                            std=statistics.pstdev(amounts), median=statistics.median(amounts))

    def payee_p2m_ratio(self, payee_vpa, as_of):
        rows = self._members(self._payee_key(payee_vpa), as_of, None)
        if not rows:
            return None
        p2m_count = sum(1 for r in rows if r["txn_type"] == "P2M")
        return (p2m_count, len(rows))

    # -- pair -----------------------------------------------------------

    def pair_txn_count(self, payer_vpa, payee_vpa, as_of):
        lo, hi = _range_bounds(_epoch(as_of), None)
        return self._client.zcount(self._pair_key(payer_vpa, payee_vpa), lo, hi)

    def pair_last_txn_time(self, payer_vpa, payee_vpa, as_of):
        hi = f"({_epoch(as_of)!r}"
        entries = self._client.zrevrangebyscore(self._pair_key(payer_vpa, payee_vpa), hi, "-inf",
                                                  start=0, num=1, withscores=True)
        if not entries:
            return None
        return _from_epoch(entries[0][1])

    # -- device -----------------------------------------------------------

    def device_distinct_payers(self, device_id, as_of, window):
        rows = self._members(self._device_key(device_id), as_of, window)
        return len({r["counterpart"] for r in rows})

    def device_txn_count(self, device_id, as_of, window):
        lo, hi = _range_bounds(_epoch(as_of), window)
        return self._client.zcount(self._device_key(device_id), lo, hi)

    def device_seen_for_payer(self, payer_vpa, device_id, as_of):
        lo, hi = _range_bounds(_epoch(as_of), None)
        return self._client.zcount(self._payer_device_key(payer_vpa, device_id), lo, hi) > 0

    # -- write ------------------------------------------------------------

    def record(self, event: UPIEvent) -> None:
        ts_epoch = _epoch(event.timestamp)
        hour = event.timestamp.hour

        pipe = self._client.pipeline()
        pipe.zadd(self._payer_key(event.payer_vpa), {
            _encode(event.txn_id, event.payee_vpa, event.amount, event.txn_type,
                     event.initiation_mode, hour): ts_epoch})
        pipe.zadd(self._payee_key(event.payee_vpa), {
            _encode(event.txn_id, event.payer_vpa, event.amount, event.txn_type,
                     event.initiation_mode, hour): ts_epoch})
        pipe.zadd(self._device_key(event.device_id), {
            _encode(event.txn_id, event.payer_vpa, event.amount, event.txn_type,
                     event.initiation_mode, hour): ts_epoch})
        pipe.zadd(self._pair_key(event.payer_vpa, event.payee_vpa), {event.txn_id: ts_epoch})
        pipe.zadd(self._payer_device_key(event.payer_vpa, event.device_id), {event.txn_id: ts_epoch})
        pipe.execute()
