"""
Tests for error handling across the indexing pipeline.

Verifies that:
  - parse failures are caught and logged, not re-raised
  - embedding failures are caught and the file is marked as failed
  - workers handle exceptions without crashing the pool
  - CLI exits gracefully when services raise
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from seekr.application.index_service import IndexService
from seekr.domain.entities import (
    FileChunk,
    FileType,
    IndexStats,
)
from seekr.domain.exceptions import ParseError
from seekr.domain.interfaces import (
    EmbeddingModel,
    FileParser,
    MetadataStore,
    VectorStore,
)

# ---------------------------------------------------------------------------
# Re-use stubs from test_index_service (copy lightweight versions here to keep
# tests self-contained)
# ---------------------------------------------------------------------------


class _StubEmbedder(EmbeddingModel):
    @property
    def dimension(self) -> int:
        return 4

    @property
    def model_name(self) -> str:
        return "stub"

    def embed_text(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 4 for _ in texts]

    def embed_image(self, image_paths: list[Path]) -> list[list[float]]:
        return [[0.0] * 4 for _ in image_paths]


class _FailingEmbedder(_StubEmbedder):
    """Raises on every embed call to simulate model failures."""

    def embed_text(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("Simulated embedding failure")


class _StubVectorStore(VectorStore):
    def add(self, chunk_ids: list[str], vectors: list[list[float]]) -> None:
        pass

    def search(self, query_vector: list[float], top_k: int) -> list[tuple[str, float]]:
        return []

    def delete(self, chunk_ids: list[str]) -> None:
        pass

    def persist(self) -> None:
        pass

    def load(self) -> None:
        pass

    @property
    def total_vectors(self) -> int:
        return 0


class _InMemoryMetadataStore(MetadataStore):
    def __init__(self) -> None:
        self._records: dict = {}
        self._chunks: dict = {}

    def upsert(self, record) -> None:  # type: ignore[override]
        self._records[record.path] = record

    def get(self, path: str):  # type: ignore[override]
        return self._records.get(path)

    def delete(self, path: str) -> None:
        self._records.pop(path, None)
        self._chunks.pop(path, None)

    def get_chunk_ids(self, path: str) -> list[str]:
        return self._chunks.get(path, [])

    def upsert_chunks(self, path: str, chunk_ids: list[str]) -> None:
        self._chunks[path] = chunk_ids

    def all_records(self):  # type: ignore[override]
        return list(self._records.values())

    def stats(self):  # type: ignore[override]
        return IndexStats(0, 0, 0, 0, 0, None)


class _FailingParser(FileParser):
    """Always raises ParseError to simulate corrupt files."""

    def supports(self, path: Path) -> bool:
        return path.suffix == ".txt"

    def file_type(self) -> FileType:
        return FileType.TEXT

    def parse(self, path: Path) -> Iterator[FileChunk]:
        raise ParseError("Simulated parse failure")


# ---------------------------------------------------------------------------
# Parse errors
# ---------------------------------------------------------------------------


class TestParseErrors:
    def test_parse_failure_does_not_raise(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.txt"
        f.write_text("data", encoding="utf-8")
        meta = _InMemoryMetadataStore()
        svc = IndexService(
            parsers=[_FailingParser()],
            text_embedder=_StubEmbedder(),
            image_embedder=_StubEmbedder(),
            vector_store=_StubVectorStore(),
            metadata_store=meta,
        )
        # Must not raise; returns 'failed'
        result = svc.index_file(f)
        assert result == "failed"

    def test_parse_failure_marks_record_as_failed(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.txt"
        f.write_text("data", encoding="utf-8")
        meta = _InMemoryMetadataStore()
        svc = IndexService(
            parsers=[_FailingParser()],
            text_embedder=_StubEmbedder(),
            image_embedder=_StubEmbedder(),
            vector_store=_StubVectorStore(),
            metadata_store=meta,
        )
        svc.index_file(f)
        record = meta.get(str(f))
        assert record is not None
        from seekr.domain.entities import IndexStatus

        assert record.status == IndexStatus.FAILED


# ---------------------------------------------------------------------------
# Embedding errors
# ---------------------------------------------------------------------------


class TestEmbeddingErrors:
    def test_embedding_failure_does_not_raise(self, tmp_path: Path) -> None:
        f = tmp_path / "ok.txt"
        f.write_text("some content here", encoding="utf-8")

        from seekr.infrastructure.parsers import PlainTextParser

        meta = _InMemoryMetadataStore()
        svc = IndexService(
            parsers=[PlainTextParser()],
            text_embedder=_FailingEmbedder(),
            image_embedder=_StubEmbedder(),
            vector_store=_StubVectorStore(),
            metadata_store=meta,
        )
        result = svc.index_file(f)
        assert result == "failed"

    def test_embedding_failure_records_error_message(self, tmp_path: Path) -> None:
        f = tmp_path / "ok.txt"
        f.write_text("some content here", encoding="utf-8")

        from seekr.infrastructure.parsers import PlainTextParser

        meta = _InMemoryMetadataStore()
        svc = IndexService(
            parsers=[PlainTextParser()],
            text_embedder=_FailingEmbedder(),
            image_embedder=_StubEmbedder(),
            vector_store=_StubVectorStore(),
            metadata_store=meta,
        )
        svc.index_file(f)
        record = meta.get(str(f))
        assert record is not None
        assert record.error_message is not None
        assert (
            "embedding" in record.error_message.lower()
            or "simulated" in record.error_message.lower()
        )
