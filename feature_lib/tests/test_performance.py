"""Performance: compute_features_batch over 50,000 events must finish in
under 60 seconds -- the InMemoryHistoryStore has to be genuinely fast
(per-entity histories + bisect + incremental prefix sums), not just
correct, since Phase 2's backfill is ~317k events x ~17 store queries each.
"""
import time
from datetime import datetime, timedelta, timezone

import numpy as np

from feature_lib.compute import compute_features_batch
from feature_lib.store.in_memory import InMemoryHistoryStore

from .conftest import make_event

N_EVENTS = 50_000
N_PAYERS = 4000
N_PAYEES = 1500
N_DEVICES = 4200


def _build_events():
    rng = np.random.default_rng(0)
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)

    payers = [f"payer{i}@x" for i in range(N_PAYERS)]
    payees = [f"payee{i}@y" for i in range(N_PAYEES)]
    devices = [f"dev-{i}" for i in range(N_DEVICES)]

    events = []
    for i in range(N_EVENTS):
        ts = t0 + timedelta(seconds=int(rng.integers(0, 90 * 24 * 3600)))
        payer = payers[int(rng.integers(0, N_PAYERS))]
        payee = payees[int(rng.integers(0, N_PAYEES))]
        device = devices[int(rng.integers(0, N_DEVICES))]
        amount = float(rng.lognormal(mean=6.0, sigma=0.6))
        events.append((i, ts, payer, payee, device, amount))

    events.sort(key=lambda row: row[1])
    return [
        make_event(f"perf-{i}", ts, payer, payee, amount=amount, device_id=device)
        for i, ts, payer, payee, device, amount in events
    ]


def test_compute_features_batch_50k_events_under_60_seconds():
    events = _build_events()
    store = InMemoryHistoryStore()

    start = time.perf_counter()
    results = compute_features_batch(events, store)
    elapsed = time.perf_counter() - start

    assert len(results) == N_EVENTS
    assert elapsed < 60, f"compute_features_batch took {elapsed:.1f}s for {N_EVENTS} events (limit 60s)"
