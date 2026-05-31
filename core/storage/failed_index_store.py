"""
core/storage/failed_index_store.py

SQLite-backed store for failed_index_records.

Why SQLite
----------
The ingestion service is a single offline process.  A lightweight embedded
database is the correct choice — no separate service, no network, no
operational overhead.  SQLite gives us ACID writes, queryable state, and
a file that can be inspected directly.

Schema
------
    failed_index_records (
        id           INTEGER  PRIMARY KEY AUTOINCREMENT,
        chunk_id     TEXT     NOT NULL,
        doc_id       TEXT     NOT NULL,
        failure_mode TEXT     NOT NULL,   -- 'es_write' | 'qdrant_write' | 'both'
        error_msg    TEXT,
        attempt_count INTEGER DEFAULT 0,
        created_at   TEXT     NOT NULL,   -- ISO-8601 UTC
        last_attempt TEXT,                -- ISO-8601 UTC
        resolved     INTEGER  DEFAULT 0   -- 0=pending, 1=resolved
    )

Exposed to callers (operators, API):
  - Records are queryable by resolved status.
  - The retry command marks records resolved=1 on success.
  - The reconciliation command inserts new records for drift it detects.
  - Path configurable via RAG_FAILED_INDEX_DB env var.

Public API
----------
    store = FailedIndexStore(db_path)
    store.record_failure(chunk_id, doc_id, failure_mode, error_msg)
    pending = store.list_pending()
    store.mark_resolved(record_id)
    store.mark_attempt(record_id)
    count = store.pending_count()
"""

from __future__ import annotations

import sqlite3
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path("data/failed_index_records.db")

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS failed_index_records (
    id            INTEGER  PRIMARY KEY AUTOINCREMENT,
    chunk_id      TEXT     NOT NULL,
    doc_id        TEXT     NOT NULL,
    failure_mode  TEXT     NOT NULL,
    error_msg     TEXT,
    attempt_count INTEGER  NOT NULL DEFAULT 0,
    created_at    TEXT     NOT NULL,
    last_attempt  TEXT,
    resolved      INTEGER  NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_fir_resolved  ON failed_index_records(resolved);
CREATE INDEX IF NOT EXISTS idx_fir_chunk_id  ON failed_index_records(chunk_id);
CREATE INDEX IF NOT EXISTS idx_fir_doc_id    ON failed_index_records(doc_id);
"""


@dataclass
class FailedIndexRecord:
    id: int
    chunk_id: str
    doc_id: str
    failure_mode: str        # 'es_write' | 'qdrant_write' | 'both'
    error_msg: str | None
    attempt_count: int
    created_at: str          # ISO-8601 UTC
    last_attempt: str | None
    resolved: bool


class FailedIndexStore:
    """
    Manages the failed_index_records SQLite table.

    Parameters
    ----------
    db_path:
        Path to the SQLite file. Created (including parent dirs) if absent.
    """

    def __init__(self, db_path: str | Path = _DEFAULT_DB_PATH) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_CREATE_TABLE_SQL)
        self._conn.commit()
        logger.debug("FailedIndexStore opened: %s", self._path)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def record_failure(
        self,
        chunk_id: str,
        doc_id: str,
        failure_mode: str,
        error_msg: str | None = None,
    ) -> int:
        """
        Insert a new failure record.  Returns the new record id.
        If a pending record for this chunk_id already exists, increments
        attempt_count instead of inserting a duplicate.
        """
        now = _utcnow()
        # Check for existing pending record
        row = self._conn.execute(
            "SELECT id, attempt_count FROM failed_index_records "
            "WHERE chunk_id = ? AND resolved = 0",
            (chunk_id,),
        ).fetchone()

        if row:
            self._conn.execute(
                "UPDATE failed_index_records "
                "SET attempt_count = ?, last_attempt = ?, error_msg = ?, failure_mode = ? "
                "WHERE id = ?",
                (row["attempt_count"] + 1, now, error_msg, failure_mode, row["id"]),
            )
            self._conn.commit()
            return row["id"]

        cur = self._conn.execute(
            "INSERT INTO failed_index_records "
            "(chunk_id, doc_id, failure_mode, error_msg, attempt_count, created_at) "
            "VALUES (?, ?, ?, ?, 0, ?)",
            (chunk_id, doc_id, failure_mode, error_msg, now),
        )
        self._conn.commit()
        return cur.lastrowid

    def mark_resolved(self, record_id: int) -> None:
        """Mark a record as resolved (retry succeeded)."""
        self._conn.execute(
            "UPDATE failed_index_records SET resolved = 1, last_attempt = ? WHERE id = ?",
            (_utcnow(), record_id),
        )
        self._conn.commit()

    def mark_attempt(self, record_id: int, error_msg: str | None = None) -> None:
        """Increment attempt_count and update last_attempt timestamp."""
        self._conn.execute(
            "UPDATE failed_index_records "
            "SET attempt_count = attempt_count + 1, last_attempt = ?, error_msg = ? "
            "WHERE id = ?",
            (_utcnow(), error_msg, record_id),
        )
        self._conn.commit()

    def delete_resolved(self) -> int:
        """Purge all resolved records. Returns count deleted."""
        cur = self._conn.execute(
            "DELETE FROM failed_index_records WHERE resolved = 1"
        )
        self._conn.commit()
        return cur.rowcount

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def list_pending(
        self,
        limit: int = 500,
        max_attempts: int | None = None,
    ) -> list[FailedIndexRecord]:
        """
        Return pending (unresolved) records, oldest first.

        Parameters
        ----------
        limit:        Max records to return.
        max_attempts: If set, only return records with attempt_count < max_attempts.
        """
        sql = "SELECT * FROM failed_index_records WHERE resolved = 0"
        params: list = []
        if max_attempts is not None:
            sql += " AND attempt_count < ?"
            params.append(max_attempts)
        sql += " ORDER BY created_at ASC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [_row_to_record(r) for r in rows]

    def pending_count(self) -> int:
        """Return count of unresolved records."""
        row = self._conn.execute(
            "SELECT COUNT(*) as n FROM failed_index_records WHERE resolved = 0"
        ).fetchone()
        return row["n"]

    def get_pending_chunk_ids(self) -> set[str]:
        """Return set of chunk_ids with unresolved failures."""
        rows = self._conn.execute(
            "SELECT DISTINCT chunk_id FROM failed_index_records WHERE resolved = 0"
        ).fetchall()
        return {r["chunk_id"] for r in rows}

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_record(row: sqlite3.Row) -> FailedIndexRecord:
    return FailedIndexRecord(
        id=row["id"],
        chunk_id=row["chunk_id"],
        doc_id=row["doc_id"],
        failure_mode=row["failure_mode"],
        error_msg=row["error_msg"],
        attempt_count=row["attempt_count"],
        created_at=row["created_at"],
        last_attempt=row["last_attempt"],
        resolved=bool(row["resolved"]),
    )
