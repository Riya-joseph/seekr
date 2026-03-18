"""
Tests for file parsers in seekr.infrastructure.parsers.

Parsers are responsible for extracting text chunks from different file types.
These tests use temporary files to exercise real parsing behaviour without
depending on the rest of the indexing pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from seekr.domain.entities import FileType
from seekr.infrastructure.parsers import (
    CodeParser,
    ImageParser,
    PDFParser,
    PlainTextParser,
)

# ---------------------------------------------------------------------------
# PlainTextParser
# ---------------------------------------------------------------------------


class TestPlainTextParser:
    @pytest.fixture()
    def parser(self) -> PlainTextParser:
        return PlainTextParser()

    def test_supports_txt(self, parser: PlainTextParser) -> None:
        assert parser.supports(Path("notes.txt"))

    def test_supports_md(self, parser: PlainTextParser) -> None:
        assert parser.supports(Path("README.md"))

    def test_supports_json(self, parser: PlainTextParser) -> None:
        assert parser.supports(Path("config.json"))

    def test_does_not_support_py(self, parser: PlainTextParser) -> None:
        assert not parser.supports(Path("script.py"))

    def test_does_not_support_jpg(self, parser: PlainTextParser) -> None:
        assert not parser.supports(Path("photo.jpg"))

    def test_file_type_is_text(self, parser: PlainTextParser) -> None:
        assert parser.file_type() == FileType.TEXT

    def test_parse_simple_text(self, parser: PlainTextParser, tmp_path: Path) -> None:
        f = tmp_path / "hello.txt"
        f.write_text("Hello, world!", encoding="utf-8")
        chunks = list(parser.parse(f))
        assert len(chunks) >= 1
        combined = " ".join(c.content for c in chunks)
        assert "Hello, world!" in combined

    def test_parse_empty_file_produces_no_chunks(
        self, parser: PlainTextParser, tmp_path: Path
    ) -> None:
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        assert list(parser.parse(f)) == []

    def test_chunks_have_correct_file_type(self, parser: PlainTextParser, tmp_path: Path) -> None:
        f = tmp_path / "doc.txt"
        f.write_text("Some content here.\n" * 100, encoding="utf-8")
        for chunk in parser.parse(f):
            assert chunk.file_type == FileType.TEXT

    def test_chunks_carry_file_path(self, parser: PlainTextParser, tmp_path: Path) -> None:
        f = tmp_path / "doc.txt"
        f.write_text("content", encoding="utf-8")
        for chunk in parser.parse(f):
            assert chunk.file_path == str(f)


# ---------------------------------------------------------------------------
# CodeParser
# ---------------------------------------------------------------------------


class TestCodeParser:
    @pytest.fixture()
    def parser(self) -> CodeParser:
        return CodeParser()

    def test_supports_py(self, parser: CodeParser) -> None:
        assert parser.supports(Path("main.py"))

    def test_supports_js(self, parser: CodeParser) -> None:
        assert parser.supports(Path("app.js"))

    def test_supports_ts(self, parser: CodeParser) -> None:
        assert parser.supports(Path("index.ts"))

    def test_does_not_support_txt(self, parser: CodeParser) -> None:
        assert not parser.supports(Path("readme.txt"))

    def test_file_type_is_code(self, parser: CodeParser) -> None:
        assert parser.file_type() == FileType.CODE

    def test_parse_python_code(self, parser: CodeParser, tmp_path: Path) -> None:
        f = tmp_path / "script.py"
        f.write_text("def hello():\n    return 'world'\n", encoding="utf-8")
        chunks = list(parser.parse(f))
        assert len(chunks) >= 1
        assert any("hello" in c.content for c in chunks)


# ---------------------------------------------------------------------------
# PDFParser  — only smoke-tests (pypdf may not be installed in all envs)
# ---------------------------------------------------------------------------


class TestPDFParser:
    @pytest.fixture()
    def parser(self) -> PDFParser:
        return PDFParser()

    def test_supports_pdf_when_pypdf_available(self, parser: PDFParser) -> None:
        try:
            import pypdf  # noqa: F401

            assert parser.supports(Path("doc.pdf"))
        except ImportError:
            pytest.skip("pypdf not installed")

    def test_does_not_support_txt(self, parser: PDFParser) -> None:
        assert not parser.supports(Path("doc.txt"))

    def test_file_type_is_document(self, parser: PDFParser) -> None:
        assert parser.file_type() == FileType.DOCUMENT


# ---------------------------------------------------------------------------
# ImageParser
# ---------------------------------------------------------------------------


class TestImageParser:
    @pytest.fixture()
    def parser(self) -> ImageParser:
        return ImageParser()

    def test_supports_jpg(self, parser: ImageParser) -> None:
        assert parser.supports(Path("photo.jpg"))

    def test_supports_png(self, parser: ImageParser) -> None:
        assert parser.supports(Path("screenshot.png"))

    def test_does_not_support_pdf(self, parser: ImageParser) -> None:
        assert not parser.supports(Path("doc.pdf"))

    def test_file_type_is_image(self, parser: ImageParser) -> None:
        assert parser.file_type() == FileType.IMAGE

    def test_parse_produces_single_chunk(self, parser: ImageParser, tmp_path: Path) -> None:
        try:
            from PIL import Image

            img_path = tmp_path / "test.png"
            img = Image.new("RGB", (10, 10), color=(255, 0, 0))
            img.save(img_path)
            chunks = list(parser.parse(img_path))
            assert len(chunks) == 1
            assert chunks[0].file_type == FileType.IMAGE
        except ImportError:
            pytest.skip("Pillow not installed")
