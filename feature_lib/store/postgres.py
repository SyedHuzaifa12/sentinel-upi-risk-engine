"""Postgres-backed HistoryStore -- the offline/training implementation.

Every query is a parameterized `WHERE ... AND timestamp < %s`, which is
where PIT correctness actually lives for this store (never `<=`). Not
imported by `feature_lib.store`'s `__init__.py`, so importing feature_lib
never requires psycopg to be installed -- import this module directly:

    from feature_lib.store.postgres import PostgresHistoryStore
"""
from datetime import timedelta
from typing import Optional

import psycopg

from feature_lib.event import UPIEvent

from .base import AmountStats, HistoricalTxn, HistoryStore

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS feature_lib_events (
    txn_id TEXT PRIMARY KEY,
    "timestamp" TIMESTAMPTZ NOT NULL,
    payer_vpa TEXT NOT NULL,
    payee_vpa TEXT NOT NULL,
    amount DOUBLE PRECISION NOT NULL,
    txn_type TEXT NOT NULL,
    initiation_mode TEXT NOT NULL,
    device_id TEXT NOT NULL,
    payer_bank TEXT NOT NULL,
    payee_bank TEXT NOT NULL,
    payer_account_age_days INTEGER NOT NULL,
    label_is_fraud BOOLEAN NOT NULL,
    label_typology TEXT
);
CREATE INDEX IF NOT EXISTS ix_events_payer_ts ON feature_lib_events (payer_vpa, "timestamp");
CREATE INDEX IF NOT EXISTS ix_events_payee_ts ON feature_lib_events (payee_vpa, "timestamp");
CREATE INDEX IF NOT EXISTS ix_events_device_ts ON feature_lib_events (device_id, "timestamp");
CREATE INDEX IF NOT EXISTS ix_events_pair_ts ON feature_lib_events (payer_vpa, payee_vpa, "timestamp");
"""


def _window_start(as_of, window: Optional[timedelta]):
    return None if window is None else as_of - window


class PostgresHistoryStore(HistoryStore):
    def __init__(self, dsn: str, reset: bool = False):
        self._conn = psycopg.connect(dsn, autocommit=True)
        if reset:
            self._conn.execute("DROP TABLE IF EXISTS feature_lib_events")
        self._conn.execute(_CREATE_TABLE_SQL)

    def close(self):
        self._conn.close()

    def _where_payer(self, payer_vpa, as_of, window):
        start = _window_start(as_of, window)
        if start is None:
            return "payer_vpa = %s AND \"timestamp\" < %s", (payer_vpa, as_of)
        return "payer_vpa = %s AND \"timestamp\" >= %s AND \"timestamp\" < %s", (payer_vpa, start, as_of)

    def _where_payee(self, payee_vpa, as_of, window):
        start = _window_start(as_of, window)
        if start is None:
            return "payee_vpa = %s AND \"timestamp\" < %s", (payee_vpa, as_of)
        return "payee_vpa = %s AND \"timestamp\" >= %s AND \"timestamp\" < %s", (payee_vpa, start, as_of)

    def _where_device(self, device_id, as_of, window):
        start = _window_start(as_of, window)
        if start is None:
            return "device_id = %s AND \"timestamp\" < %s", (device_id, as_of)
        return "device_id = %s AND \"timestamp\" >= %s AND \"timestamp\" < %s", (device_id, start, as_of)

    # -- payer --------------------------------------------------------

    def payer_txn_count(self, payer_vpa, as_of, window):
        where, params = self._where_payer(payer_vpa, as_of, window)
        row = self._conn.execute(f"SELECT COUNT(*) FROM feature_lib_events WHERE {where}", params).fetchone()
        return row[0]

    def payer_distinct_payees(self, payer_vpa, as_of, window):
        where, params = self._where_payer(payer_vpa, as_of, window)
        row = self._conn.execute(
            f"SELECT COUNT(DISTINCT payee_vpa) FROM feature_lib_events WHERE {where}", params).fetchone()
        return row[0]

    def payer_last_txn(self, payer_vpa, as_of):
        row = self._conn.execute(
            "SELECT \"timestamp\", amount, payee_vpa FROM feature_lib_events "
            "WHERE payer_vpa = %s AND \"timestamp\" < %s ORDER BY \"timestamp\" DESC LIMIT 1",
            (payer_vpa, as_of),
        ).fetchone()
        if row is None:
            return None
        return HistoricalTxn(timestamp=row[0], amount=row[1], payee_vpa=row[2])

    def payer_amount_stats(self, payer_vpa, as_of, window):
        where, params = self._where_payer(payer_vpa, as_of, window)
        row = self._conn.execute(
            f"SELECT COUNT(*), AVG(amount), STDDEV_POP(amount), "
            f"PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount) "
            f"FROM feature_lib_events WHERE {where}", params).fetchone()
        count = row[0]
        if count == 0:
            return AmountStats(count=0, mean=None, std=None, median=None)
        return AmountStats(count=count, mean=float(row[1]), std=float(row[2] or 0.0), median=float(row[3]))

    def payer_hour_histogram(self, payer_vpa, as_of, window):
        where, params = self._where_payer(payer_vpa, as_of, window)
        rows = self._conn.execute(
            f"SELECT EXTRACT(HOUR FROM \"timestamp\")::int, COUNT(*) "
            f"FROM feature_lib_events WHERE {where} GROUP BY 1", params).fetchall()
        return dict(rows)

    def payer_new_payee_txn_ratio(self, payer_vpa, as_of, window):
        where, params = self._where_payer(payer_vpa, as_of, window)
        total = self._conn.execute(f"SELECT COUNT(*) FROM feature_lib_events WHERE {where}", params).fetchone()[0]
        if total == 0:
            return (0, 0)
        # A payee counts as "new" at row r if it never appeared for this
        # payer strictly before r's own timestamp.
        new_count = self._conn.execute(
            f"""
            SELECT COUNT(*) FROM feature_lib_events e
            WHERE {where}
              AND NOT EXISTS (
                SELECT 1 FROM feature_lib_events prior
                WHERE prior.payer_vpa = e.payer_vpa AND prior.payee_vpa = e.payee_vpa
                  AND prior."timestamp" < e."timestamp"
              )
            """, params).fetchone()[0]
        return (new_count, total)

    def payer_collect_request_ratio(self, payer_vpa, as_of, window):
        where, params = self._where_payer(payer_vpa, as_of, window)
        row = self._conn.execute(
            f"SELECT COUNT(*) FILTER (WHERE initiation_mode = 'COLLECT_REQUEST'), COUNT(*) "
            f"FROM feature_lib_events WHERE {where}", params).fetchone()
        return (row[0], row[1])

    # -- payee --------------------------------------------------------

    def payee_txn_count(self, payee_vpa, as_of):
        row = self._conn.execute(
            "SELECT COUNT(*) FROM feature_lib_events WHERE payee_vpa = %s AND \"timestamp\" < %s",
            (payee_vpa, as_of)).fetchone()
        return row[0]

    def payee_first_seen(self, payee_vpa, as_of):
        row = self._conn.execute(
            "SELECT MIN(\"timestamp\") FROM feature_lib_events WHERE payee_vpa = %s AND \"timestamp\" < %s",
            (payee_vpa, as_of)).fetchone()
        return row[0]

    def payee_distinct_payers(self, payee_vpa, as_of, window):
        where, params = self._where_payee(payee_vpa, as_of, window)
        row = self._conn.execute(
            f"SELECT COUNT(DISTINCT payer_vpa) FROM feature_lib_events WHERE {where}", params).fetchone()
        return row[0]

    def payee_amount_stats(self, payee_vpa, as_of, window=None):
        where, params = self._where_payee(payee_vpa, as_of, window)
        row = self._conn.execute(
            f"SELECT COUNT(*), AVG(amount), STDDEV_POP(amount), "
            f"PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount) "
            f"FROM feature_lib_events WHERE {where}", params).fetchone()
        count = row[0]
        if count == 0:
            return AmountStats(count=0, mean=None, std=None, median=None)
        return AmountStats(count=count, mean=float(row[1]), std=float(row[2] or 0.0), median=float(row[3]))

    def payee_p2m_ratio(self, payee_vpa, as_of):
        row = self._conn.execute(
            "SELECT COUNT(*) FILTER (WHERE txn_type = 'P2M'), COUNT(*) "
            "FROM feature_lib_events WHERE payee_vpa = %s AND \"timestamp\" < %s",
            (payee_vpa, as_of)).fetchone()
        if row[1] == 0:
            return None
        return (row[0], row[1])

    # -- pair -----------------------------------------------------------

    def pair_txn_count(self, payer_vpa, payee_vpa, as_of):
        row = self._conn.execute(
            "SELECT COUNT(*) FROM feature_lib_events "
            "WHERE payer_vpa = %s AND payee_vpa = %s AND \"timestamp\" < %s",
            (payer_vpa, payee_vpa, as_of)).fetchone()
        return row[0]

    def pair_last_txn_time(self, payer_vpa, payee_vpa, as_of):
        row = self._conn.execute(
            "SELECT MAX(\"timestamp\") FROM feature_lib_events "
            "WHERE payer_vpa = %s AND payee_vpa = %s AND \"timestamp\" < %s",
            (payer_vpa, payee_vpa, as_of)).fetchone()
        return row[0]

    # -- device -----------------------------------------------------------

    def device_distinct_payers(self, device_id, as_of, window):
        where, params = self._where_device(device_id, as_of, window)
        row = self._conn.execute(
            f"SELECT COUNT(DISTINCT payer_vpa) FROM feature_lib_events WHERE {where}", params).fetchone()
        return row[0]

    def device_txn_count(self, device_id, as_of, window):
        where, params = self._where_device(device_id, as_of, window)
        row = self._conn.execute(f"SELECT COUNT(*) FROM feature_lib_events WHERE {where}", params).fetchone()
        return row[0]

    def device_seen_for_payer(self, payer_vpa, device_id, as_of):
        row = self._conn.execute(
            "SELECT 1 FROM feature_lib_events "
            "WHERE payer_vpa = %s AND device_id = %s AND \"timestamp\" < %s LIMIT 1",
            (payer_vpa, device_id, as_of)).fetchone()
        return row is not None

    # -- write ------------------------------------------------------------

    def record(self, event: UPIEvent) -> None:
        self._conn.execute(
            "INSERT INTO feature_lib_events "
            "(txn_id, \"timestamp\", payer_vpa, payee_vpa, amount, txn_type, initiation_mode, "
            " device_id, payer_bank, payee_bank, payer_account_age_days, label_is_fraud, label_typology) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (txn_id) DO NOTHING",
            (event.txn_id, event.timestamp, event.payer_vpa, event.payee_vpa, event.amount,
             event.txn_type, event.initiation_mode, event.device_id, event.payer_bank,
             event.payee_bank, event.payer_account_age_days, event.label_is_fraud, event.label_typology),
        )
