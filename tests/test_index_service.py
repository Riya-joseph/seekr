"""
Tests for the indexing pipeline in seekr.application.index_service.

These tests use in-memory stubs for all infrastructure dependencies so they
run fast with no real files, models, or databases.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from seekr.application.index_service import IndexService
from seekr.domain.entities import (
    FileChunk,
    FileRecord,
    FileType,
    IndexStats,
)
from seekr.domain.exceptions import IndexingError
from seekr.domain.interfaces import (
    EmbeddingModel,
    FileParser,
    MetadataStore,
    VectorStore,
)

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubEmbedder(EmbeddingModel):
    @property
    def dimension(self) -> int:
        return 4

    @property
    def model_name(self) -> str:
        return "stub"

    def embed_text(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    def embed_image(self, image_paths: list[Path]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in image_paths]


class _StubVectorStore(VectorStore):
    def __init__(self) -> None:
        self.added: list[tuple[list[str], list[list[float]]]] = []
        self.deleted: list[list[str]] = []
        self.persisted = False

    def add(self, chunk_ids: list[str], vectors: list[list[float]]) -> None:
        self.added.append((chunk_ids, vectors))

    def search(self, query_vector: list[float], top_k: int) -> list[tuple[str, float]]:
        return []

    def delete(self, chunk_ids: list[str]) -> None:
        self.deleted.append(chunk_ids)

    def persist(self) -> None:
        self.persisted = True

    def load(self) -> None:
        pass

    @property
    def total_vectors(self) -> int:
        return 0


class _InMemoryMetadataStore(MetadataStore):
    def __init__(self) -> None:
        self._records: dict[str, FileRecord] = {}
        self._chunks: dict[str, list[str]] = {}

    def upsert(self, record: FileRecord) -> None:
        self._records[record.path] = record

    def get(self, path: str) -> FileRecord | None:
        return self._records.get(path)

    def delete(self, path: str) -> None:
        self._records.pop(path, None)
        self._chunks.pop(path, None)

    def get_chunk_ids(self, path: str) -> list[str]:
        return self._chunks.get(path, [])

    def upsert_chunks(self, path: str, chunk_ids: list[str]) -> None:
        self._chunks[path] = chunk_ids

    def all_records(self) -> list[FileRecord]:
        return list(self._records.values())

    def stats(self) -> IndexStats:
        return IndexStats(
            total_files=len(self._records),
            total_chunks=0,
            text_files=0,
            image_files=0,
            index_size_bytes=0,
            last_updated=None,
        )


class _StubParser(FileParser):
    def __init__(self, ext: str = ".txt", file_type: FileType = FileType.TEXT) -> None:
        self._ext = ext
        self._type = file_type

    def supports(self, path: Path) -> bool:
        return path.suffix == self._ext

    def file_type(self) -> FileType:
        return self._type

    def parse(self, path: Path) -> Iterator[FileChunk]:
        yield FileChunk(
            file_path=str(path),
            chunk_index=0,
            content="stub content",
            start_char=0,
            end_char=12,
            file_type=self._type,
        )


def _make_service(
    parsers: list[FileParser] | None = None,
    meta: MetadataStore | None = None,
    vec: VectorStore | None = None,
) -> tuple[IndexService, _InMemoryMetadataStore, _StubVectorStore]:
    metadata = meta or _InMemoryMetadataStore()
    vector = vec or _StubVectorStore()
    svc = IndexService(
        parsers=parsers or [_StubParser()],
        text_embedder=_StubEmbedder(),
        image_embedder=_StubEmbedder(),
        vector_store=vector,
        metadata_store=metadata,
    )
    return svc, metadata, vector


# ---------------------------------------------------------------------------
# index_path
# ---------------------------------------------------------------------------


class TestIndexPath:
    def test_indexes_supported_files(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.txt"
        f.write_text("hello world", encoding="utf-8")

        svc, _meta, _ = _make_service()
        counts = svc.index_path(tmp_path)
        assert counts.get("indexed", 0) >= 1

    def test_skips_unchanged_files(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.txt"
        f.write_text("hello world", encoding="utf-8")

        svc, _meta, _ = _make_service()
        svc.index_path(tmp_path)
        counts2 = svc.index_path(tmp_path)
        assert counts2.get("skipped", 0) >= 1

    def test_raises_on_nonexistent_path(self) -> None:
        svc, _, _ = _make_service()
        with pytest.raises(IndexingError):
            svc.index_path(Path("/nonexistent/path/xyz"))

    def test_unsupported_files_are_not_indexed(self, tmp_path: Path) -> None:
        f = tmp_path / "file.xyz_unknown"
        f.write_text("data", encoding="utf-8")

        svc, meta, _ = _make_service(parsers=[_StubParser(".txt")])
        svc.index_path(tmp_path)
        # No .txt file present → nothing indexed
        assert meta.all_records() == []


# ---------------------------------------------------------------------------
# remove_file
# ---------------------------------------------------------------------------


class TestRemoveFile:
    def test_removes_record_and_vectors(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.txt"
        f.write_text("hello world", encoding="utf-8")

        svc, meta, _vec = _make_service()
        svc.index_path(tmp_path)
        assert meta.get(str(f)) is not None

        svc.remove_file(f)
        assert meta.get(str(f)) is None

    def test_remove_nonexistent_file_is_safe(self, tmp_path: Path) -> None:
        svc, _, _ = _make_service()
        # Should not raise
        svc.remove_file(tmp_path / "ghost.txt")


# ---------------------------------------------------------------------------
# dry_run
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_dry_run_new_files_appear_in_to_index(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("content", encoding="utf-8")
        (tmp_path / "b.txt").write_text("content", encoding="utf-8")

        svc, _, _ = _make_service()
        result = svc.dry_run(tmp_path)
        assert len(result["to_index"]) == 2
        assert len(result["already_indexed"]) == 0
        assert "estimated_chunks" in result
        # Legacy key still present for backward compat
        assert result["file_paths"] == result["to_index"]

    def test_dry_run_already_indexed_files_are_separated(self, tmp_path: Path) -> None:
        """Files already in the index with matching SHA should appear in already_indexed."""
        f = tmp_path / "a.txt"
        f.write_text("hello seekr", encoding="utf-8")

        svc, meta, _ = _make_service()
        # Index the file for real first so the metadata store knows about it.
        svc.index_path(tmp_path)
        assert len(meta.all_records()) == 1

        # dry_run on the same dir should now classify the file as already_indexed.
        result = svc.dry_run(tmp_path)
        assert len(result["to_index"]) == 0
        assert len(result["already_indexed"]) == 1

    def test_dry_run_changed_file_appears_in_to_index(self, tmp_path: Path) -> None:
        """A file whose content changed since indexing must appear in to_index."""
        f = tmp_path / "note.txt"
        f.write_text("original content", encoding="utf-8")

        svc, _, _ = _make_service()
        svc.index_path(tmp_path)

        # Modify the file after indexing.
        f.write_text("completely different content", encoding="utf-8")

        result = svc.dry_run(tmp_path)
        assert len(result["to_index"]) == 1
        assert len(result["already_indexed"]) == 0

    def test_dry_run_does_not_write_metadata(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("content", encoding="utf-8")

        svc, meta, _ = _make_service()
        svc.dry_run(tmp_path)
        assert meta.all_records() == []


# ---------------------------------------------------------------------------
# prune_path
# ---------------------------------------------------------------------------


class TestPrunePath:
    def test_prune_removes_all_files_under_root(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "a.txt").write_text("hello", encoding="utf-8")
        (sub / "b.txt").write_text("world", encoding="utf-8")

        svc, meta, _ = _make_service()
        svc.index_path(sub)
        assert len(meta.all_records()) == 2

        result = svc.prune_path(sub)
        assert result["removed"] == 2
        assert meta.all_records() == []

    def test_prune_returns_zero_when_nothing_indexed(self, tmp_path: Path) -> None:
        svc, _, _ = _make_service()
        result = svc.prune_path(tmp_path)
        assert result["removed"] == 0
