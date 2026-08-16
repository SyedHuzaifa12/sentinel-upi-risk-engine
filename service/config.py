"""Env-driven settings for the scoring service. No Django dependency --
this package must run standalone (uvicorn service.main:app)."""
import os

STORE_BACKEND = os.getenv("STORE_BACKEND", "memory")
SEED_EVENTS = os.getenv("SEED_EVENTS", "false").lower() in ("1", "true", "yes")
REDIS_URL = os.getenv("REDIS_URL")

# Same env var name Django uses (backend/config/settings.py, via
# dj-database-url) -- one Postgres instance, one variable name. Used both for
# STORE_BACKEND=postgres (feature history) and, unconditionally, for the
# decision log (decisions are always written to Postgres regardless of which
# HistoryStore backend is active).
DATABASE_URL = os.getenv("DATABASE_URL")

# Measured, not aspirational -- see PROGRESS.md's Phase 4 benchmark table for
# how close service/benchmark.py actually gets to this.
LATENCY_BUDGET_P99_MS = 100.0
