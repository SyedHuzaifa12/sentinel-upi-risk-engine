"""Every file in sql/ must execute without error against the live schema --
these are the raw-SQL artifacts backend/users/views/monitoring.py loads and
runs via django.db.connection.cursor() (see "do not use the ORM here" in
CLAUDE.md's Phase 6 section). This test exercises the exact same execution
path monitoring_dashboard() uses, against an empty-but-correctly-shaped
decisions/users_reviewlabel schema, so a broken query fails CI immediately
rather than surfacing as a 500 on the live dashboard.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from django.db import connection
from django.test import TestCase

from decisionlog.schema import _CREATE_TABLE_SQL  # noqa: E402

SQL_DIR = Path(__file__).resolve().parents[3] / "sql"


class SqlFilesExecuteTests(TestCase):
    def setUp(self):
        with connection.cursor() as cursor:
            cursor.execute(_CREATE_TABLE_SQL)

    def test_every_sql_file_executes_without_error(self):
        sql_files = sorted(SQL_DIR.glob("*.sql"))
        self.assertTrue(sql_files, f"no .sql files found under {SQL_DIR}")

        for path in sql_files:
            with self.subTest(file=path.name):
                with connection.cursor() as cursor:
                    cursor.execute(path.read_text())
                    # Every query in sql/ is a SELECT (monitoring-only reads) --
                    # fetchall() must not raise even against zero matching rows.
                    cursor.fetchall()
