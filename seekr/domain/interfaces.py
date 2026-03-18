"""
Domain interfaces (ports) for Seekr.

These abstract base classes define the contracts that infrastructure
implementations must satisfy. The application layer depends only on
these interfaces — never on concrete implementations.

Architecture note: This is the "ports" side of ports-and-adapters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Optional, Sequence

from seekr.domain.entities import (
    FileChunk,
    FileRecord,
    FileType,
    IndexStats,
    IndexTask,
    QueueStats,
    SearchResult,
)


class EmbeddingModel(ABC):
    """
    Port for embedding a piece of text or image into a dense vector.

    Concrete implementations live in infrastructure/.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the embedding vectors produced."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier."""
        ...

    @abstractmethod
    def embed_text(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of text strings.

        Args:
            texts: Non-empty list of strings to embed.

        Returns:
            List of float vectors, one per input string.
        """
        ...

    @abstractmethod
    def embed_image(self, image_paths: list[Path]) -> list[list[float]]:
        """
        Embed a batch of images.

        Args:
            image_paths: Paths to image files (PNG, JPEG, etc.)

        Returns:
            List of float vectors in the same embedding space as embed_text.
        """
        ...


class VectorStore(ABC):
    """
    Port for a vector similarity index.

    Implementations must support incremental add/delete and
    nearest-neighbour search by cosine similarity.
    """

    @abstractmethod
    def add(self, chunk_ids: list[str], vectors: list[list[float]]) -> None:
        """
        Add or update vectors in the store.

        If a chunk_id already exists it should be overwritten.

        Args:
            chunk_ids: Unique identifiers for each vector.
            vectors:   Corresponding embedding vectors.
        """
        ...

    @abstractmethod
    def search(self, query_vector: list[float], top_k: int) -> list[tuple[str, float]]:
        """
        Find the top-k most similar vectors to the query.

        Args:
            query_vector: Query embedding.
            top_k:        Maximum number of results to return.

        Returns:
            List of (chunk_id, similarity_score) tuples, sorted descending.
        """
        ...

    @abstractmethod
    def delete(self, chunk_ids: list[str]) -> None:
        """
        Remove vectors from the store.

        IDs that do not exist should be silently ignored.
        """
        ...

    @abstractmethod
    def persist(self) -> None:
        """Flush in-memory state to disk."""
        ...

    @abstractmethod
    def load(self) -> None:
        """Load persisted state from disk into memory."""
        ...

    @property
    @abstractmethod
    def total_vectors(self) -> int:
        """Number of vectors currently stored."""
        ...


class FileParser(ABC):
    """
    Port for extracting text chunks from a file.

    Implementations handle specific file types (plain text, PDF, images, …).
    """

    @abstractmethod
    def supports(self, path: Path) -> bool:
        """Return True if this parser can handle the given file."""
        ...

    @abstractmethod
    def parse(self, path: Path) -> Iterator[FileChunk]:
        """
        Parse a file and yield content chunks.

        Args:
            path: Absolute path to the file.

        Yields:
            FileChunk objects in order.
        """
        ...

    @abstractmethod
    def file_type(self) -> FileType:
        """The FileType this parser produces."""
        ...


class MetadataStore(ABC):
    """
    Port for persisting file metadata alongside the vector store.

    Keeps track of sha256 hashes, timestamps, and index status so
    that the indexer can skip unchanged files.
    """

    @abstractmethod
    def upsert(self, record: FileRecord) -> None:
        """Insert or replace a file record."""
        ...

    @abstractmethod
    def get(self, path: str) -> Optional[FileRecord]:
        """Retrieve the record for a given file path, or None."""
        ...

    @abstractmethod
    def delete(self, path: str) -> None:
        """Remove the record for a given file path."""
        ...

    @abstractmethod
    def get_chunk_ids(self, path: str) -> list[str]:
        """Return all chunk_ids associated with a file path."""
        ...

    @abstractmethod
    def upsert_chunks(self, path: str, chunk_ids: list[str]) -> None:
        """Store the chunk_id mapping for a file."""
        ...

    @abstractmethod
    def all_records(self) -> list[FileRecord]:
        """Return all stored file records."""
        ...

    @abstractmethod
    def stats(self) -> IndexStats:
        """Return aggregate statistics about the index."""
        ...


class IndexQueue(ABC):
    """
    Port for the background indexing task queue.

    Implementations persist tasks (e.g. in SQLite) so that workers can
    process them asynchronously. Used when indexing in background mode.
    """

    @abstractmethod
    def enqueue_file(self, file_path: str, file_hash: str) -> int:
        """
        Add a file to the indexing queue.

        Returns the task id.
        """
        ...

    @abstractmethod
    def get_pending_tasks(self, limit: int) -> list[IndexTask]:
        """Return up to *limit* tasks with status 'pending'."""
        ...

    @abstractmethod
    def mark_processing(self, task_id: int) -> None:
        """Mark a task as being processed."""
        ...

    @abstractmethod
    def mark_done(self, task_id: int) -> None:
        """Mark a task as completed successfully."""
        ...

    @abstractmethod
    def mark_failed(self, task_id: int) -> None:
        """Mark a task as failed (e.g. parse error)."""
        ...

    @abstractmethod
    def get_stats(self) -> QueueStats:
        """Return queue progress statistics."""
        ...


class FileWatcher(ABC):
    """
    Port for monitoring a directory tree for file system events.
    """

    @abstractmethod
    def start(
        self,
        paths: list[Path],
        on_created: ...,
        on_modified: ...,
        on_deleted: ...,
    ) -> None:
        """
        Begin watching the given paths.

        Callbacks receive a single Path argument.
        """
        ...

    @abstractmethod
    def stop(self) -> None:
        """Stop the watcher and release resources."""
        ...
