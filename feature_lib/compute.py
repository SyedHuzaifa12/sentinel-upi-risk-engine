"""compute_features(), compute_features_batch(), and is_cold() -- the public
entry points training, the FastAPI service, and the stream worker all
import, unmodified.
"""
from .event import UPIEvent
from .registry import BASE_REGISTRY
from .store.base import HistoryStore
from .vector import FeatureVector


def is_cold(event: UPIEvent, store: HistoryStore) -> bool:
    return store.payee_txn_count(event.payee_vpa, as_of=event.timestamp) == 0


def compute_features(event: UPIEvent, store: HistoryStore) -> FeatureVector:
    """Pure: never mutates the store, never reads a row with
    timestamp >= event.timestamp (enforced inside each HistoryStore
    implementation's own query/filter logic -- see store/base.py)."""
    values = {}
    for spec in BASE_REGISTRY:
        value, is_missing = spec.compute(event, store)
        values[spec.name] = value
        if spec.can_be_missing:
            values[f"{spec.name}_is_missing"] = is_missing

    return FeatureVector(txn_id=event.txn_id, is_cold=is_cold(event, store), values=values)


def compute_features_batch(events: list[UPIEvent], store: HistoryStore) -> list[FeatureVector]:
    """Walks events in timestamp order, computing features for event i
    using only events 0..i-1. This is how the training set is built and is
    where most people accidentally leak -- so the ordering below is the
    entire contract:

        compute THEN record, never the other way around.

    Do not reorder these two lines, and do not batch/parallelize across
    events (each event's features depend on every prior event having
    already been recorded).
    """
    results = []
    for event in events:
        results.append(compute_features(event, store))
        store.record(event)
    return results
