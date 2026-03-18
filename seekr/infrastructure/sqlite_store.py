"""
SQLiteMetadataStore — SQLite-backed implementation of the MetadataStore port.

Schema:
  files(path TEXT PK, sha256 TEXT, file_type TEXT, size_bytes INT,
        modified_at TEXT, indexed_at TEXT, chunk_count INT,
        status TEXT, error_message TEXT)
  chunks(path TEXT, chunk_id TEXT, PRIMARY KEY(path, chunk_id))

All datetime objects are stored as ISO-8601 UTC strings.
Thread-safe via a per-connection threading.local and WAL journal mode.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from seekr.domain.entities import FileRecord, FileType, IndexStats, IndexStatus
from seekr.domain.exceptions import StoreError
from seekr.domain.interfaces import MetadataStore

logger = logging.getLogger(__name__)

_DDL = """\
CREATE TABLE IF NOT EXISTS files (
    path          TEXT PRIMARY KEY,
    sha256        TEXT NOT NULL,
    file_type     TEXT NOT NULL,
    size_bytes    INTEGER NOT NULL,
    modified_at   TEXT NOT NULL,
    indexed_at    TEXT NOT NULL,
    chunk_count   INTEGER NOT NULL DEFAULT 0,
    status        TEXT NOT NULL DEFAULT 'indexed',
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
    path     TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    PRIMARY KEY (path, chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);
"""


def _dt_to_str(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _str_to_dt(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


class SQLiteMetadataStore(MetadataStore):
    """
    Persistent file metadata store backed by SQLite.

    A single DB file is used; WAL mode ensures concurrent readers
    don't block the writer.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        # Initialise schema on construction
        conn = self._conn()
        conn.executescript(_DDL)
        conn.commit()
        logger.debug("SQLite metadata store initialised at %s", db_path)

    # ------------------------------------------------------------------
    # MetadataStore interface
    # ------------------------------------------------------------------

    def upsert(self, record: FileRecord) -> None:
        conn = self._conn()
        conn.execute(
            """
            INSERT INTO files
                (path, sha256, file_type, size_bytes, modified_at,
                 indexed_at, chunk_count, status, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                sha256        = excluded.sha256,
                file_type     = excluded.file_type,
                size_bytes    = excluded.size_bytes,
                modified_at   = excluded.modified_at,
                indexed_at    = excluded.indexed_at,
                chunk_count   = excluded.chunk_count,
                status        = excluded.status,
                error_message = excluded.error_message
            """,
            (
                record.path,
                record.sha256,
                record.file_type.name,
                record.size_bytes,
                _dt_to_str(record.modified_at),
                _dt_to_str(record.indexed_at),
                record.chunk_count,
                record.status.value,
                record.error_message,
            ),
        )
        conn.commit()

    def get(self, path: str) -> Optional[FileRecord]:
        conn = self._conn()
        row = conn.execute(
            "SELECT * FROM files WHERE path = ?", (path,)
        ).fetchone()
        return self._row_to_record(row) if row else None

    def delete(self, path: str) -> None:
        conn = self._conn()
        conn.execute("DELETE FROM files WHERE path = ?", (path,))
        conn.execute("DELETE FROM chunks WHERE path = ?", (path,))
        conn.commit()

    def get_chunk_ids(self, path: str) -> list[str]:
        conn = self._conn()
        rows = conn.execute(
            "SELECT chunk_id FROM chunks WHERE path = ?", (path,)
        ).fetchall()
        return [r[0] for r in rows]

    def upsert_chunks(self, path: str, chunk_ids: list[str]) -> None:
        conn = self._conn()
        # Clear existing mappings for this file
        conn.execute("DELETE FROM chunks WHERE path = ?", (path,))
        conn.executemany(
            "INSERT OR IGNORE INTO chunks (path, chunk_id) VALUES (?, ?)",
            [(path, cid) for cid in chunk_ids],
        )
        conn.commit()

    def all_records(self) -> list[FileRecord]:
        conn = self._conn()
        rows = conn.execute("SELECT * FROM files ORDER BY indexed_at DESC").fetchall()
        return [self._row_to_record(r) for r in rows]

    def stats(self) -> IndexStats:
        conn = self._conn()
        total_files = conn.execute(
            "SELECT COUNT(*) FROM files WHERE status = 'indexed'"
        ).fetchone()[0]
        total_chunks = conn.execute(
            "SELECT COALESCE(SUM(chunk_count), 0) FROM files WHERE status = 'indexed'"
        ).fetchone()[0]
        text_files = conn.execute(
            "SELECT COUNT(*) FROM files WHERE file_type != 'IMAGE' AND status = 'indexed'"
        ).fetchone()[0]
        image_files = conn.execute(
            "SELECT COUNT(*) FROM files WHERE file_type = 'IMAGE' AND status = 'indexed'"
        ).fetchone()[0]
        last_row = conn.execute(
            "SELECT MAX(indexed_at) FROM files"
        ).fetchone()[0]
        last_updated = _str_to_dt(last_row) if last_row else None

        # Approximate size: count total characters in chunk content
        index_size = self._db_path.stat().st_size if self._db_path.exists() else 0

        return IndexStats(
            total_files=total_files,
            total_chunks=total_chunks,
            text_files=text_files,
            image_files=image_files,
            index_size_bytes=index_size,
            last_updated=last_updated,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        """Return a thread-local SQLite connection."""
        if not hasattr(self._local, "conn"):
            conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> FileRecord:
        return FileRecord(
            path=row["path"],
            sha256=row["sha256"],
            file_type=FileType[row["file_type"]],
            size_bytes=row["size_bytes"],
            modified_at=_str_to_dt(row["modified_at"]),
            indexed_at=_str_to_dt(row["indexed_at"]),
            chunk_count=row["chunk_count"],
            status=IndexStatus(row["status"]),
            error_message=row["error_message"],
        )
