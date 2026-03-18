"""
IndexService — application-layer orchestrator for file indexing.

Responsibilities:
  - Walk a directory and decide which files need (re-)indexing.
  - Delegate parsing to FileParser implementations.
  - Delegate embedding to the EmbeddingModel.
  - Write vectors to VectorStore and metadata to MetadataStore.
  - Compute sha256 hashes to detect unchanged files (skip them).

This class has zero imports from FAISS, SentenceTransformers, or any
infrastructure library. All dependencies are injected via interfaces.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path

from seekr.config.settings import MAX_CHARS_PER_FILE, MAX_CHUNKS_PER_FILE
from seekr.domain.entities import FileChunk, FileRecord, FileType, IndexStatus
from seekr.domain.exceptions import IndexingError, ParseError
from seekr.domain.interfaces import (
    EmbeddingModel,
    FileParser,
    IndexQueue,
    MetadataStore,
    VectorStore,
)
from seekr.domain.patterns import is_ignored as _is_path_ignored_by_pattern
from seekr.domain.patterns import matches_pattern

logger = logging.getLogger(__name__)

# Files larger than this are chunked more aggressively
_MAX_BATCH_VECTORS = 64

# Always skip these when indexing a directory, even if no ignore_patterns are passed.
# Prevents accidentally indexing whole machine or pulling in deps by default.
_MINIMUM_IGNORE: frozenset[str] = frozenset(
    {
        ".git",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
    }
)


def _sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(65536), b""):
                h.update(block)
    except OSError as exc:
        raise IndexingError(f"Cannot read file for hashing: {path}") from exc
    return h.hexdigest()


class IndexService:
    """
    Application service for indexing files and directories.

    All heavy infrastructure objects (parsers, embedding model, stores)
    are injected; this class contains only orchestration logic.
    """

    def __init__(
        self,
        parsers: Sequence[FileParser],
        text_embedder: EmbeddingModel,
        image_embedder: EmbeddingModel,
        vector_store: VectorStore,
        metadata_store: MetadataStore,
        progress_callback: Callable[[str, int, int], None] | None = None,
        image_vector_store: VectorStore | None = None,
        queue: IndexQueue | None = None,
    ) -> None:
        """
        Args:
            parsers:           Ordered list of FileParser implementations.
            text_embedder:     Embedder for text chunks.
            image_embedder:    Embedder for image files (must share embedding
                               space with text_embedder for cross-modal search).
            vector_store:      Where text/code chunk vectors are stored (e.g. 384-dim).
            metadata_store:    Where file metadata is stored.
            progress_callback: Optional fn(current_file, done, total) for UIs.
            image_vector_store: If set, image vectors (e.g. 512-dim CLIP) are stored here.
                               When None, images use vector_store (e.g. when use_clip_for_text).
            queue:             If set, index_path/index_file only enqueue tasks; workers do the work.
        """
        self._parsers = list(parsers)
        self._text_embedder = text_embedder
        self._image_embedder = image_embedder
        self._vector_store = vector_store
        self._image_vector_store = image_vector_store
        self._metadata_store = metadata_store
        self._progress_callback = progress_callback
        self._queue = queue

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_path(
        self,
        root: Path,
        ignore_patterns: set[str] | None = None,
    ) -> dict[str, int]:
        """
        Recursively index all supported files under *root*.

        Paths whose path components match any of *ignore_patterns* (e.g. "node_modules")
        are skipped. Use infrastructure.ignore.load_ignore_patterns() to build the set.

        When a queue is configured, only enqueues tasks (scan + enqueue) and returns
        counts with key "queued". Otherwise performs synchronous indexing and returns
        indexed, skipped, failed.
        """
        root = root.resolve()
        if not root.exists():
            raise IndexingError(f"Path does not exist: {root}")

        files = self._collect_files(root, ignore_patterns=ignore_patterns)
        logger.info("Found %d candidate files under %s", len(files), root)

        if self._queue is not None:
            counts = self._enqueue_path(files)
            logger.info(
                "Queued %d tasks (skipped %d)", counts.get("queued", 0), counts.get("skipped", 0)
            )
            return counts

        sync_counts: dict[str, int] = {"indexed": 0, "skipped": 0, "failed": 0}
        for i, path in enumerate(files):
            if self._progress_callback:
                self._progress_callback(str(path), i + 1, len(files))
            result = self._index_single(path)
            sync_counts[result] += 1

        self._vector_store.persist()
        if self._image_vector_store is not None:
            self._image_vector_store.persist()
        logger.info(
            "Indexing complete — indexed=%d skipped=%d failed=%d",
            sync_counts["indexed"],
            sync_counts["skipped"],
            sync_counts["failed"],
        )
        return sync_counts

    def index_file(
        self,
        path: Path,
        ignore_patterns: set[str] | None = None,
    ) -> str:
        """
        Index a single file.

        When a queue is configured, enqueues the file and returns 'queued'.
        Otherwise returns one of: 'indexed', 'skipped', 'failed'.

        Args:
            path:            Path to the file to index.
            ignore_patterns: Optional set of path-component patterns to skip.
                             Checked here so that workers honouring a queued
                             task still respect the exclusions that were active
                             when the task was originally submitted.
        """
        path = path.resolve()
        if ignore_patterns and _is_path_ignored_by_pattern(path, ignore_patterns):
            logger.debug("Skipping excluded file: %s", path)
            return "skipped"
        if self._queue is not None:
            return self._enqueue_single(path)
        result = self._index_single(path)
        self._vector_store.persist()
        if self._image_vector_store is not None:
            self._image_vector_store.persist()
        return result

    def get_record(self, path: str) -> FileRecord | None:
        """Return the metadata record for a path, or None."""
        return self._metadata_store.get(path)

    def dry_run(
        self,
        root: Path,
        ignore_patterns: set[str] | None = None,
    ) -> dict[str, int | list[str]]:
        """
        Walk the tree, classify each file, and report what would actually change.

        Files are split into three buckets:
        - ``to_index``:       new files not yet in the index, or files whose content
                              has changed since they were last indexed.
        - ``already_indexed``: files whose SHA-256 matches the stored record — a real
                              ``seekr index`` run would skip these entirely.
        - ``unsupported``:    files with no matching parser (always ignored).

        No indexing, embedding, or writes are performed.

        Returns a dict with keys:
          ``to_index``        - list of path strings that would be processed
          ``already_indexed`` - list of path strings that would be skipped
          ``estimated_chunks``- estimated new chunks for ``to_index`` files only
        """
        root = root.resolve()
        if not root.exists():
            raise IndexingError(f"Path does not exist: {root}")

        files = self._collect_files(root, ignore_patterns=ignore_patterns)

        to_index: list[str] = []
        already_indexed: list[str] = []
        estimated_chunks = 0
        chars_per_chunk = 2000

        for path in files:
            parser = self._find_parser(path)
            if parser is None:
                continue

            # Check if the file is already up-to-date in the index.
            try:
                sha = _sha256(path)
            except IndexingError:
                # Unreadable file — treat as needing indexing so the user sees it.
                to_index.append(str(path))
                continue

            existing = self._metadata_store.get(str(path))
            if existing and existing.sha256 == sha and existing.status == IndexStatus.INDEXED:
                already_indexed.append(str(path))
                continue

            # New or changed file.
            to_index.append(str(path))
            if parser.file_type() == FileType.IMAGE:
                estimated_chunks += 1
            else:
                try:
                    size = path.stat().st_size
                except OSError:
                    size = 0
                chars = min(size, MAX_CHARS_PER_FILE)
                n = math.ceil(chars / chars_per_chunk) if chars else 1
                estimated_chunks += min(n, MAX_CHUNKS_PER_FILE)

        return {
            "to_index": to_index,
            "already_indexed": already_indexed,
            "estimated_chunks": estimated_chunks,
            # Legacy key — kept so any caller that used "file_paths" still works.
            "file_paths": to_index,
        }

    def remove_file(self, path: Path, persist: bool = True) -> None:
        """
        Remove a file's vectors and metadata from the index.

        Called when a watched file is deleted from disk, or by prune_path.

        Args:
            path:    Path to the file to remove.
            persist: If True (default), flush vector stores to disk immediately.
                     Pass False when removing many files in a loop (prune_path)
                     so that persist is called once at the end instead of once
                     per file.
        """
        path_str = str(path.resolve())
        chunk_ids = self._metadata_store.get_chunk_ids(path_str)
        record = self._metadata_store.get(path_str)
        if chunk_ids:
            if (
                record is not None
                and record.file_type == FileType.IMAGE
                and self._image_vector_store is not None
            ):
                self._image_vector_store.delete(chunk_ids)
            else:
                self._vector_store.delete(chunk_ids)
            logger.debug("Removed %d vectors for %s", len(chunk_ids), path_str)

        self._metadata_store.delete(path_str)
        logger.info("Removed file from index: %s", path_str)

        if persist:
            self._vector_store.persist()
            if self._image_vector_store is not None:
                self._image_vector_store.persist()

    def prune_path(self, root: Path) -> dict[str, int]:
        """
        Remove from the index all files under *root* (and root itself if a file).

        Deletes their chunk IDs from both vector stores and metadata so you can
        un-index a subtree (e.g. ~/projects/repo-a) without re-indexing everything else.

        Returns a summary dict with key "removed" (number of files removed).
        """
        root = root.resolve()
        to_remove: list[Path] = []
        if root.is_file():
            if self._metadata_store.get(str(root)) is not None:
                to_remove.append(root)
        else:
            for record in self._metadata_store.all_records():
                p = Path(record.path)
                if p == root or p.is_relative_to(root):
                    to_remove.append(p)
        for path in to_remove:
            self.remove_file(path, persist=False)
        self._vector_store.persist()
        if self._image_vector_store is not None:
            self._image_vector_store.persist()
        logger.info("Pruned %d files under %s", len(to_remove), root)
        return {"removed": len(to_remove)}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_files(
        self,
        root: Path,
        ignore_patterns: set[str] | None = None,
    ) -> list[Path]:
        """
        Walk the directory tree and return files that have a matching parser.

        Uses os.walk with topdown=True so ignored and hidden directories are
        pruned *before* recursing into them.  This is critical for performance:
        rglob("*") visits every entry (including node_modules, .git, etc.) and
        only filters afterwards, which can mean millions of filesystem calls on
        a typical Downloads or home directory.
        """
        files: list[Path] = []
        if root.is_file():
            if self._find_parser(root):
                files.append(root)
            return files

        effective = ignore_patterns if ignore_patterns is not None else _MINIMUM_IGNORE

        for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
            current = Path(dirpath)

            # Prune subdirectories in-place so os.walk never descends into them.
            # This is the key optimisation: a single ignored dir (e.g. node_modules
            # with 100k files) is excluded with one comparison rather than 100k.
            # Directories are matched with all pattern types (exact, glob, extension).
            dirnames[:] = [
                d
                for d in dirnames
                if not d.startswith(".")  # hidden dirs
                and not any(matches_pattern(d, p) for p in effective)  # ignored dirs
            ]

            for filename in filenames:
                if filename.startswith("."):  # hidden files
                    continue
                if any(matches_pattern(filename, p) for p in effective):  # all pattern types
                    continue
                path = current / filename
                if self._find_parser(path):
                    files.append(path)

        return sorted(files)

    @staticmethod
    def _is_ignored(path: Path, root: Path, patterns: set[str]) -> bool:
        """True if any path component under root is in patterns."""
        try:
            rel = path.relative_to(root)
        except ValueError:
            return False
        return any(part in patterns for part in rel.parts)

    @staticmethod
    def _is_hidden(path: Path) -> bool:
        """Return True if any component of the path starts with '.'."""
        return any(part.startswith(".") for part in path.parts)

    def _find_parser(self, path: Path) -> FileParser | None:
        """Return the first parser that supports this file, or None."""
        for parser in self._parsers:
            if parser.supports(path):
                return parser
        return None

    def _enqueue_single(self, path: Path) -> str:
        """
        Enqueue a single file for background indexing (idempotency: skip if already indexed).
        Returns 'queued' or 'skipped'.
        """
        assert self._queue is not None, "_enqueue_single requires a configured queue"
        parser = self._find_parser(path)
        if parser is None:
            return "skipped"
        try:
            sha = _sha256(path)
        except IndexingError as exc:
            logger.warning("Hash failed for %s: %s", path, exc)
            return "skipped"
        existing = self._metadata_store.get(str(path))
        if existing and existing.sha256 == sha and existing.status == IndexStatus.INDEXED:
            logger.debug("Unchanged, skipping: %s", path)
            return "skipped"
        self._queue.enqueue_file(str(path), sha)
        return "queued"

    def _enqueue_path(self, files: list[Path]) -> dict[str, int]:
        """Enqueue all files for background indexing; return counts (queued, skipped)."""
        counts: dict[str, int] = {"queued": 0, "skipped": 0}
        for i, path in enumerate(files):
            if self._progress_callback:
                self._progress_callback(str(path), i + 1, len(files))
            result = self._enqueue_single(path)
            counts[result] = counts.get(result, 0) + 1
        return counts

    def _index_single(self, path: Path) -> str:
        """
        Index one file.  Returns 'indexed' | 'skipped' | 'failed'.
        """
        parser = self._find_parser(path)
        if parser is None:
            return "skipped"

        try:
            sha = _sha256(path)
        except IndexingError as exc:
            logger.warning("Hash failed for %s: %s", path, exc)
            return "failed"

        # Check if already up-to-date
        existing = self._metadata_store.get(str(path))
        if existing and existing.sha256 == sha and existing.status == IndexStatus.INDEXED:
            logger.debug("Unchanged, skipping: %s", path)
            return "skipped"

        # Remove stale vectors if the file was previously indexed
        if existing:
            stale_ids = self._metadata_store.get_chunk_ids(str(path))
            if stale_ids:
                if existing.file_type == FileType.IMAGE and self._image_vector_store is not None:
                    self._image_vector_store.delete(stale_ids)
                else:
                    self._vector_store.delete(stale_ids)

        try:
            chunks = list(parser.parse(path))
        except ParseError as exc:
            logger.warning("Parse error for %s: %s", path, exc)
            self._metadata_store.upsert(
                FileRecord(
                    path=str(path),
                    sha256=sha,
                    file_type=parser.file_type(),
                    size_bytes=path.stat().st_size,
                    modified_at=datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc),
                    indexed_at=datetime.now(tz=timezone.utc),
                    chunk_count=0,
                    status=IndexStatus.FAILED,
                    error_message=str(exc),
                )
            )
            return "failed"

        if not chunks:
            logger.debug("No chunks extracted from %s", path)
            return "skipped"

        try:
            chunk_ids = self._embed_and_store(chunks, parser.file_type(), path)
        except Exception as exc:
            logger.error("Embedding/store failed for %s: %s", path, exc, exc_info=True)
            self._metadata_store.upsert(
                FileRecord(
                    path=str(path),
                    sha256=sha,
                    file_type=parser.file_type(),
                    size_bytes=path.stat().st_size,
                    modified_at=datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc),
                    indexed_at=datetime.now(tz=timezone.utc),
                    chunk_count=0,
                    status=IndexStatus.FAILED,
                    error_message=str(exc),
                )
            )
            return "failed"

        stat = path.stat()
        record = FileRecord(
            path=str(path),
            sha256=sha,
            file_type=parser.file_type(),
            size_bytes=stat.st_size,
            modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            indexed_at=datetime.now(tz=timezone.utc),
            chunk_count=len(chunk_ids),
            status=IndexStatus.INDEXED,
        )
        self._metadata_store.upsert(record)
        self._metadata_store.upsert_chunks(str(path), chunk_ids)
        logger.debug("Indexed %s (%d chunks)", path, len(chunk_ids))
        return "indexed"

    def _embed_and_store(
        self, chunks: list[FileChunk], file_type: FileType, path: Path
    ) -> list[str]:
        """
        Embed all chunks and write to the vector store.

        Defensive cap: if chunk count exceeds MAX_CHUNKS_PER_FILE, truncate
        so large files do not dominate the index. Only embedded chunks are stored.
        """
        chunk_ids: list[str] = []
        texts: list[str] = []
        ids_for_batch: list[str] = []

        def _flush_text_batch() -> None:
            if not texts:
                return
            vectors = self._text_embedder.embed_text(texts)
            self._vector_store.add(ids_for_batch[:], vectors)
            texts.clear()
            ids_for_batch.clear()

        if file_type != FileType.IMAGE and len(chunks) > MAX_CHUNKS_PER_FILE:
            original = len(chunks)
            chunks = chunks[:MAX_CHUNKS_PER_FILE]
            logger.info(
                "Chunk cap applied file=%s original_chunks=%d kept_chunks=%d",
                path,
                original,
                len(chunks),
            )

        if file_type == FileType.IMAGE:
            vectors = self._image_embedder.embed_image([path])
            cid = f"{path}::0"
            store = (
                self._image_vector_store
                if self._image_vector_store is not None
                else self._vector_store
            )
            store.add([cid], vectors)
            chunk_ids.append(cid)
        else:
            for chunk in chunks:
                texts.append(chunk.content)
                ids_for_batch.append(chunk.chunk_id)
                chunk_ids.append(chunk.chunk_id)
                if len(texts) >= _MAX_BATCH_VECTORS:
                    _flush_text_batch()
            _flush_text_batch()

        return chunk_ids
