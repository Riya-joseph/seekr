"""
Tests for the text chunking logic in seekr.infrastructure.parsers.

Chunking is a critical pipeline stage: correct behaviour ensures that
  - large files are split into retrievable pieces
  - the chunk cap prevents index explosion
  - overlapping windows preserve context at boundaries
  - empty / whitespace-only inputs produce no chunks
"""

from __future__ import annotations

from seekr.domain.entities import FileChunk, FileType
from seekr.infrastructure.parsers import _chunk_text

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunks(text: str, max_chunks: int = 200) -> list[FileChunk]:
    return list(_chunk_text(text, path="test.txt", file_type=FileType.TEXT, max_chunks=max_chunks))


# ---------------------------------------------------------------------------
# Basic behaviour
# ---------------------------------------------------------------------------


class TestChunkText:
    def test_empty_string_produces_no_chunks(self) -> None:
        assert _chunks("") == []

    def test_whitespace_only_produces_no_chunks(self) -> None:
        assert _chunks("   \n\t  ") == []

    def test_short_text_contains_content(self) -> None:
        """Short text always appears in the first chunk regardless of overlap sliding."""
        chunks = _chunks("Hello world")
        assert len(chunks) >= 1
        # The very first chunk must contain the full short text
        assert chunks[0].content == "Hello world"
        assert chunks[0].chunk_index == 0

    def test_chunk_ids_are_unique(self) -> None:
        text = " ".join(["word"] * 10_000)
        chunks = _chunks(text)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_index_is_sequential(self) -> None:
        text = " ".join(["word"] * 10_000)
        chunks = _chunks(text)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_chunk_content_is_not_empty(self) -> None:
        text = "a " * 5000
        for chunk in _chunks(text):
            assert chunk.content.strip()

    def test_file_type_propagated(self) -> None:
        chunks = list(_chunk_text("some code", path="foo.py", file_type=FileType.CODE))
        assert all(c.file_type == FileType.CODE for c in chunks)

    def test_file_path_propagated(self) -> None:
        chunks = list(
            _chunk_text("hello world", path="/some/path/file.txt", file_type=FileType.TEXT)
        )
        assert all(c.file_path == "/some/path/file.txt" for c in chunks)


# ---------------------------------------------------------------------------
# Chunk cap
# ---------------------------------------------------------------------------


class TestChunkCap:
    def test_chunk_cap_is_respected(self) -> None:
        # 50_000 words will produce many more than 5 chunks without a cap
        text = " ".join(["word"] * 50_000)
        chunks = _chunks(text, max_chunks=5)
        assert len(chunks) <= 5

    def test_default_cap_is_respected(self) -> None:
        text = " ".join(["word"] * 200_000)
        chunks = _chunks(text)
        assert len(chunks) <= 200

    def test_cap_of_one_produces_single_chunk(self) -> None:
        text = " ".join(["word"] * 10_000)
        chunks = _chunks(text, max_chunks=1)
        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# Boundary / overlap
# ---------------------------------------------------------------------------


class TestChunkBoundaries:
    def test_start_char_of_first_chunk_is_zero(self) -> None:
        text = "hello world " * 500
        chunks = _chunks(text)
        assert chunks[0].start_char == 0

    def test_end_char_greater_than_start_char(self) -> None:
        text = "hello world " * 500
        for chunk in _chunks(text):
            assert chunk.end_char > chunk.start_char

    def test_chunk_id_format(self) -> None:
        chunks = _chunks("hello world", max_chunks=1)
        assert chunks[0].chunk_id == "test.txt::0"
