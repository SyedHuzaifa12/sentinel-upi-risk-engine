# feature_lib

The single shared, point-in-time-correct feature layer for the UPI risk engine.
Imported unmodified by training, the FastAPI serving layer, and the stream worker
(per `CLAUDE.md`'s hard rule: one feature implementation, never duplicated).

- `event.py` -- the canonical `UPIEvent` schema.
- `registry.py` -- `BASE_REGISTRY` (37 computed features) and the derived
  `REGISTRY`/`COLD_FEATURES`/`WARM_ONLY_FEATURES`/`ALL_FEATURES`.
- `compute.py` -- `compute_features`, `compute_features_batch`, `is_cold`.
- `features/` -- the five feature groups (A: context, B: payer behaviour,
  C: payee newness, D: payee track record, E: device).
- `store/` -- `HistoryStore` ABC plus `InMemoryHistoryStore` (fast, bisect +
  prefix sums), `PostgresHistoryStore`, `RedisHistoryStore`.

## Known gap: Postgres parity test is currently skipped

Neither Postgres nor Redis is reachable in the development environment this
package was built in (no `psql`/`redis-cli`, nothing listening on 5432/6379, no
`DATABASE_URL`/`REDIS_URL` configured). `PostgresHistoryStore` and
`RedisHistoryStore` are real, complete implementations — not stubs — but
`feature_lib/tests/test_parity.py` (InMemory vs. Postgres agreement on identical
input) currently **skips** rather than passing, because it can't reach a
database to run against.

**This must actually pass — not just skip — before Phase 5 ships.** Wire up a
reachable Postgres instance (set `TEST_DATABASE_URL`) and confirm
`pytest feature_lib/tests` shows the parity test passing, not skipped, before
treating the Postgres store as production-ready.
