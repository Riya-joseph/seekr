"""
Domain entities for Seekr.

Pure Python dataclasses with no external dependencies.
These are the core business objects shared across all layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Optional


class FileType(Enum):
    """Supported file types for indexing."""
    TEXT = auto()
    IMAGE = auto()
    CODE = auto()
    DOCUMENT = auto()
    UNKNOWN = auto()


class IndexStatus(Enum):
    """Status of an indexing operation for a file record."""
    PENDING = "pending"
    INDEXED = "indexed"
    FAILED = "failed"
    DELETED = "deleted"
    SKIPPED = "skipped"


class TaskStatus(Enum):
    """Status of a background indexing task in the queue."""
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


@dataclass(frozen=True)
class FileChunk:
    """
    A chunk of content extracted from a file.

    Large files are split into overlapping chunks so that
    embeddings capture fine-grained semantic regions.
    """
    file_path: str
    chunk_index: int
    content: str
    start_char: int
    end_char: int
    file_type: FileType

    @property
    def chunk_id(self) -> str:
        """Unique identifier for this chunk."""
        return f"{self.file_path}::{self.chunk_index}"


@dataclass(frozen=True)
class FileRecord:
    """
    Metadata record for an indexed file.

    Stored in SQLite; the vector store references records by chunk_id.
    """
    path: str
    sha256: str
    file_type: FileType
    size_bytes: int
    modified_at: datetime
    indexed_at: datetime
    chunk_count: int
    status: IndexStatus = IndexStatus.INDEXED
    error_message: Optional[str] = None


@dataclass
class SearchResult:
    """
    A single result from a semantic search query.

    Includes the source chunk and its similarity score.
    """
    chunk_id: str
    file_path: str
    chunk_index: int
    score: float                    # cosine similarity [0, 1]
    snippet: str                    # excerpt of the matched chunk
    file_type: FileType
    modified_at: Optional[datetime] = None



@dataclass
class IndexStats:
    """Aggregate statistics about the current index state."""
    total_files: int
    total_chunks: int
    text_files: int
    image_files: int
    index_size_bytes: int
    last_updated: Optional[datetime]
    watch_paths: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class IndexTask:
    """A single task in the indexing queue."""
    id: int
    file_path: str
    file_hash: str
    status: TaskStatus
    created_at: datetime
    updated_at: datetime


@dataclass
class QueueStats:
    """Progress statistics for the background indexing queue."""
    total: int
    completed: int
    processing: int
    pending: int
    failed: int

    @property
    def progress_pct(self) -> float:
        """Progress as a percentage (0–100)."""
        if self.total == 0:
            return 0.0
        return 100.0 * self.completed / self.total
