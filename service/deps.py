"""HistoryStore factory + FastAPI dependency to fetch the request-scoped
store from app state.

STORE_BACKEND=postgres/redis are wired against the real constructors in
feature_lib.store.{postgres,redis_store}, but only "memory" is exercised in
this environment (Postgres is still unreachable here -- see PROGRESS.md).
This is infrastructure-ready, not infrastructure-verified, and is documented
as such rather than implied to have been tested.
"""
from fastapi import Request

from feature_lib.store.base import HistoryStore
from feature_lib.store.in_memory import InMemoryHistoryStore

from . import config


def build_store() -> HistoryStore:
    backend = config.STORE_BACKEND
    if backend == "memory":
        return InMemoryHistoryStore()
    if backend == "postgres":
        from feature_lib.store.postgres import PostgresHistoryStore
        if not config.POSTGRES_DSN:
            raise RuntimeError("STORE_BACKEND=postgres requires POSTGRES_DSN to be set.")
        return PostgresHistoryStore(config.POSTGRES_DSN)
    if backend == "redis":
        from feature_lib.store.redis_store import RedisHistoryStore
        if not config.REDIS_URL:
            raise RuntimeError("STORE_BACKEND=redis requires REDIS_URL to be set.")
        return RedisHistoryStore(config.REDIS_URL)
    raise ValueError(f"Unknown STORE_BACKEND={backend!r} (expected memory|postgres|redis)")


def get_store(request: Request) -> HistoryStore:
    return request.app.state.store
