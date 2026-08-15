"""The HistoryStore abstract interface.

Every read method takes `as_of: datetime` and is contractually PIT-correct:
implementations must only ever consider rows with `timestamp < as_of`,
never `<= as_of`. This is enforced by construction in each implementation's
query/filter logic (e.g. `WHERE timestamp < %s` in Postgres, a strict
`bisect_left` cutoff in-memory) rather than by a bolted-on runtime check —
see feature_lib/tests/test_pit_correctness.py for the behavioral proof.

`record(event)` is the only write method. `compute_features_batch` in
feature_lib/compute.py is the only place that should call both
`compute_features` and `record` together, and it must always compute
before recording (see that module's docstring).
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from feature_lib.event import UPIEvent


@dataclass
class AmountStats:
    count: int
    mean: Optional[float]
    std: Optional[float]
    median: Optional[float]


@dataclass
class HistoricalTxn:
    timestamp: datetime
    amount: float
    payee_vpa: Optional[str] = None
    payer_vpa: Optional[str] = None


class HistoryStore(ABC):
    @abstractmethod
    def payer_txn_count(self, payer_vpa: str, as_of: datetime, window: Optional[timedelta]) -> int:
        ...

    @abstractmethod
    def payer_distinct_payees(self, payer_vpa: str, as_of: datetime, window: Optional[timedelta]) -> int:
        ...

    @abstractmethod
    def payer_last_txn(self, payer_vpa: str, as_of: datetime) -> Optional[HistoricalTxn]:
        ...

    @abstractmethod
    def payer_amount_stats(self, payer_vpa: str, as_of: datetime, window: Optional[timedelta]) -> AmountStats:
        ...

    @abstractmethod
    def payer_hour_histogram(self, payer_vpa: str, as_of: datetime, window: Optional[timedelta]) -> dict:
        """Returns {hour_of_day (0-23): count} over the window."""
        ...

    @abstractmethod
    def payer_new_payee_txn_ratio(self, payer_vpa: str, as_of: datetime, window: Optional[timedelta]) -> tuple:
        """Returns (new_payee_txn_count, total_txn_count) over the window."""
        ...

    @abstractmethod
    def payer_collect_request_ratio(self, payer_vpa: str, as_of: datetime, window: Optional[timedelta]) -> tuple:
        """Returns (collect_request_txn_count, total_txn_count) over the window."""
        ...

    @abstractmethod
    def payee_txn_count(self, payee_vpa: str, as_of: datetime) -> int:
        """All-time count. Used directly by is_cold()."""
        ...

    @abstractmethod
    def payee_first_seen(self, payee_vpa: str, as_of: datetime) -> Optional[datetime]:
        ...

    @abstractmethod
    def payee_distinct_payers(self, payee_vpa: str, as_of: datetime, window: Optional[timedelta]) -> int:
        ...

    @abstractmethod
    def payee_amount_stats(self, payee_vpa: str, as_of: datetime, window: Optional[timedelta] = None) -> AmountStats:
        ...

    @abstractmethod
    def payee_p2m_ratio(self, payee_vpa: str, as_of: datetime) -> Optional[tuple]:
        """Returns (p2m_txn_count, total_txn_count), all-time, or None if never seen."""
        ...

    @abstractmethod
    def pair_txn_count(self, payer_vpa: str, payee_vpa: str, as_of: datetime) -> int:
        ...

    @abstractmethod
    def pair_last_txn_time(self, payer_vpa: str, payee_vpa: str, as_of: datetime) -> Optional[datetime]:
        ...

    @abstractmethod
    def device_distinct_payers(self, device_id: str, as_of: datetime, window: Optional[timedelta]) -> int:
        ...

    @abstractmethod
    def device_txn_count(self, device_id: str, as_of: datetime, window: Optional[timedelta]) -> int:
        ...

    @abstractmethod
    def device_seen_for_payer(self, payer_vpa: str, device_id: str, as_of: datetime) -> bool:
        ...

    @abstractmethod
    def record(self, event: UPIEvent) -> None:
        ...
