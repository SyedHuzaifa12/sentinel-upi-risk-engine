"""Schema for the immutable `decisions` table.

Same mechanism `feature_lib/store/postgres.py`'s `PostgresHistoryStore` already
uses for `feature_lib_events` -- an idempotent `CREATE TABLE IF NOT EXISTS`
executed by the writer itself, not a Django migration and not a separate SQL
init script. `decisions` isn't Django-ORM-owned data, so a migration would be
the wrong tool; a second ad-hoc mechanism alongside the one Phase 1 already
established would just be inconsistent.

Append-only is enforced for real, at the database level: BEFORE UPDATE/DELETE
triggers raise on any attempt to mutate or remove a row. This is what makes a
decision an audit record rather than a convention someone could quietly break.
"""
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS decisions (
    id BIGSERIAL PRIMARY KEY,
    txn_id TEXT NOT NULL UNIQUE,
    event JSONB NOT NULL,
    feature_snapshot JSONB NOT NULL,
    risk_score DOUBLE PRECISION NOT NULL,
    raw_score DOUBLE PRECISION NOT NULL,
    is_cold BOOLEAN NOT NULL,
    action TEXT NOT NULL,
    risk_tier TEXT NOT NULL,
    reason_codes JSONB NOT NULL,
    model_version TEXT NOT NULL,
    thresholds_version TEXT NOT NULL,
    feature_lib_version TEXT NOT NULL,
    latency_ms DOUBLE PRECISION NOT NULL,
    scored_at TIMESTAMPTZ NOT NULL,
    source TEXT NOT NULL CHECK (source IN ('api', 'worker', 'replay'))
);

-- decisions(scored_at): the phase brief calls this "decisions(created_at)" --
-- scored_at IS that timestamp (when the decision was made, not a generic row-
-- creation time); this index serves time-ordered audit/monitoring queries.
CREATE INDEX IF NOT EXISTS ix_decisions_scored_at ON decisions (scored_at);
-- decisions(action): serves "how many BLOCK/REVIEW/... today" monitoring queries.
CREATE INDEX IF NOT EXISTS ix_decisions_action ON decisions (action);
-- decisions(risk_score): serves threshold-curve / distribution analysis queries.
CREATE INDEX IF NOT EXISTS ix_decisions_risk_score ON decisions (risk_score);

CREATE OR REPLACE FUNCTION decisions_prevent_mutation() RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'decisions is append-only: % is not allowed', TG_OP;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER decisions_no_update
BEFORE UPDATE ON decisions
FOR EACH ROW EXECUTE FUNCTION decisions_prevent_mutation();

CREATE OR REPLACE TRIGGER decisions_no_delete
BEFORE DELETE ON decisions
FOR EACH ROW EXECUTE FUNCTION decisions_prevent_mutation();
"""


def ensure_schema(conn) -> None:
    """Idempotent: safe to call on every process startup (mirrors
    PostgresHistoryStore's own __init__ behavior)."""
    conn.execute(_CREATE_TABLE_SQL)
