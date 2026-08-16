"""Parity: InMemory and Postgres stores return identical vectors for the
same input. Skips (with a printed reason) if TEST_DATABASE_URL is unset or
the connection fails -- useful for running this file without Docker up.
Passes for real (not skipped) against a live Postgres container (Phase 5) --
see docker-compose.yml and PROGRESS.md's Phase 5 section.

Float comparisons tolerate only IEEE-754-noise-level disagreement (see
conftest.py's FLOAT_REL_TOL/FLOAT_ABS_TOL) -- InMemoryHistoryStore's
incremental variance and Postgres's STDDEV_POP() are both correct, just
summed in a different order. Exact equality is still required everywhere
else (ints/bools/strings, NaN-vs-NaN, missingness).
"""
import os
from datetime import datetime, timedelta, timezone

import pytest

from feature_lib.compute import compute_features, compute_features_batch
from feature_lib.store.in_memory import InMemoryHistoryStore

from .conftest import FLOAT_REL_TOL, assert_vectors_equal, make_event

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _make_postgres_store():
    dsn = os.environ.get("TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("TEST_DATABASE_URL not set -- no Postgres reachable in this environment")
    try:
        from feature_lib.store.postgres import PostgresHistoryStore
        return PostgresHistoryStore(dsn, reset=True)
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"Postgres unreachable: {exc}")


def _sample_events():
    events = [
        make_event("t1", T0, "alice@x", "shop@y", amount=200.0),
        make_event("t2", T0 + timedelta(minutes=10), "bob@x", "shop@y", amount=250.0),
        make_event("t3", T0 + timedelta(minutes=20), "alice@x", "shop@y", amount=300.0),
        make_event("t4", T0 + timedelta(hours=1), "alice@x", "carol@z", amount=50.0,
                    initiation_mode="COLLECT_REQUEST"),
        make_event("t5", T0 + timedelta(hours=2), "dave@w", "shop@y", amount=400.0, device_id="dev-2"),
    ]
    target = make_event("t-target", T0 + timedelta(hours=3), "alice@x", "shop@y", amount=275.0)
    return events, target


def test_in_memory_and_postgres_agree_on_the_same_input():
    pg_store = _make_postgres_store()
    try:
        memory_store = InMemoryHistoryStore()
        events, target = _sample_events()

        compute_features_batch(events, memory_store)
        compute_features_batch(events, pg_store)

        memory_vector = compute_features(target, memory_store)
        pg_vector = compute_features(target, pg_store)

        max_rel_diff = assert_vectors_equal(memory_vector.values, pg_vector.values)
        assert memory_vector.is_cold == pg_vector.is_cold
        print(f"\nPostgres parity: PASSED. Max relative float difference observed: "
              f"{max_rel_diff:.3e} (tolerance: {FLOAT_REL_TOL:.0e})")
    finally:
        pg_store.close()
