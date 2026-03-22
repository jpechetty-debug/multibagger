"""Pytest configuration for local package imports."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data.db import db


TEST_TICKER_PATTERNS = ("TEST%", "P6%")


def _purge_runtime_test_artifacts() -> None:
    """Remove persisted test rows from runtime SQLite databases."""

    for target in ("stocks", "ops", "pit", "swing", "mb"):
        with db.connection(target) as conn:
            table_rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
            for row in table_rows:
                table_name = row["name"]
                column_rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
                columns = {column["name"] for column in column_rows}
                if "ticker" in columns:
                    conn.execute(
                        f"DELETE FROM {table_name} WHERE ticker LIKE ? OR ticker LIKE ?",
                        TEST_TICKER_PATTERNS,
                    )
            if target == "ops":
                conn.execute(
                    "DELETE FROM logs WHERE context_json LIKE ? OR context_json LIKE ? OR message LIKE ? OR message LIKE ?",
                    ("%TEST%", "%P6%", "%TEST%", "%P6%"),
                )
                conn.execute(
                    "DELETE FROM run_history WHERE command_args_json LIKE ? OR summary LIKE ? OR command_name LIKE ?",
                    ("%TEST%", "%TEST%", "%phase6%"),
                )


@pytest.fixture(scope="session", autouse=True)
def cleanup_runtime_test_artifacts() -> None:
    """Keep the shared runtime databases free of persisted test symbols."""

    _purge_runtime_test_artifacts()
    yield
    _purge_runtime_test_artifacts()
