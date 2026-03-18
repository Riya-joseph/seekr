"""
SQLiteIndexQueue — SQLite-backed implementation of the IndexQueue port.

Stores index_tasks in the same metadata.db used by SQLiteMetadataStore.
Thread-safe via thread-local connections and WAL mode.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from seekr.domain.entities import IndexTask, QueueStats, TaskStatus
from seekr.domain.interfaces import IndexQueue

logger = logging.getLogger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS index_tasks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path   TEXT NOT NULL,
    file_hash   TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_index_tasks_status ON index_tasks(status);
"""


def _dt_to_str(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _str_to_dt(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


class SQLiteIndexQueue(IndexQueue):
    """
    Persistent task queue backed by SQLite.

    Uses the same DB file as metadata (e.g. ~/.seekr/metadata.db).
    WAL mode allows concurrent read/write with the metadata store.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        conn = self._conn()
        conn.executescript(_DDL)
        conn.commit()
        logger.debug("Index queue initialised at %s", db_path)

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    def enqueue_file(self, file_path: str, file_hash: str) -> int:
        now = _dt_to_str(datetime.now(timezone.utc))
        conn = self._conn()
        cur = conn.execute(
            """
            INSERT INTO index_tasks (file_path, file_hash, status, created_at, updated_at)
            VALUES (?, ?, 'pending', ?, ?)
            """,
            (file_path, file_hash, now, now),
        )
        conn.commit()
        task_id = cur.lastrowid
        logger.debug("Enqueued task %s for %s", task_id, file_path)
        return task_id

    def get_pending_tasks(self, limit: int) -> list[IndexTask]:
        conn = self._conn()
        rows = conn.execute(
            """
            SELECT id, file_path, file_hash, status, created_at, updated_at
            FROM index_tasks
            WHERE status = 'pending'
            ORDER BY id
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [self._row_to_task(r) for r in rows]

    def mark_processing(self, task_id: int) -> None:
        now = _dt_to_str(datetime.now(timezone.utc))
        conn = self._conn()
        conn.execute(
            "UPDATE index_tasks SET status = 'processing', updated_at = ? WHERE id = ?",
            (now, task_id),
        )
        conn.commit()

    def mark_done(self, task_id: int) -> None:
        now = _dt_to_str(datetime.now(timezone.utc))
        conn = self._conn()
        conn.execute(
            "UPDATE index_tasks SET status = 'done', updated_at = ? WHERE id = ?",
            (now, task_id),
        )
        conn.commit()

    def mark_failed(self, task_id: int) -> None:
        now = _dt_to_str(datetime.now(timezone.utc))
        conn = self._conn()
        conn.execute(
            "UPDATE index_tasks SET status = 'failed', updated_at = ? WHERE id = ?",
            (now, task_id),
        )
        conn.commit()

    def get_stats(self) -> QueueStats:
        conn = self._conn()
        total = conn.execute("SELECT COUNT(*) FROM index_tasks").fetchone()[0]
        completed = conn.execute(
            "SELECT COUNT(*) FROM index_tasks WHERE status = 'done'"
        ).fetchone()[0]
        processing = conn.execute(
            "SELECT COUNT(*) FROM index_tasks WHERE status = 'processing'"
        ).fetchone()[0]
        pending = conn.execute(
            "SELECT COUNT(*) FROM index_tasks WHERE status = 'pending'"
        ).fetchone()[0]
        failed = conn.execute(
            "SELECT COUNT(*) FROM index_tasks WHERE status = 'failed'"
        ).fetchone()[0]
        return QueueStats(
            total=total,
            completed=completed,
            processing=processing,
            pending=pending,
            failed=failed,
        )

    @staticmethod
    def _row_to_task(row: sqlite3.Row) -> IndexTask:
        return IndexTask(
            id=row["id"],
            file_path=row["file_path"],
            file_hash=row["file_hash"],
            status=TaskStatus(row["status"]),
            created_at=_str_to_dt(row["created_at"]),
            updated_at=_str_to_dt(row["updated_at"]),
        )
