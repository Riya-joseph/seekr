"""
File parsers — infrastructure implementations of the FileParser port.

Each parser handles a specific set of file types and yields FileChunk objects.
Chunking respects MAX_CHARS_PER_FILE and MAX_CHUNKS_PER_FILE from config
so large files do not dominate the index.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

from seekr.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    MAX_CHARS_PER_FILE,
    MAX_CHUNKS_PER_FILE,
)
from seekr.domain.entities import FileChunk, FileType
from seekr.domain.exceptions import ParseError
from seekr.domain.interfaces import FileParser

logger = logging.getLogger(__name__)

# Approximate chars-per-token for Latin text
_CHARS_PER_TOKEN = 4
_CHUNK_SIZE_CHARS = CHUNK_SIZE * _CHARS_PER_TOKEN
_OVERLAP_SIZE_CHARS = CHUNK_OVERLAP * _CHARS_PER_TOKEN
_MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MB hard cap — refuse to open larger files


def _chunk_text(
    text: str,
    path: str,
    file_type: FileType,
    max_chunks: int = MAX_CHUNKS_PER_FILE,
) -> Iterator[FileChunk]:
    """
    Split text into overlapping chunks and yield FileChunk objects.

    Stops after max_chunks chunks so large files do not produce thousands of chunks.
    Splits on whitespace boundaries so chunks never cut a word in half.
    O(N) in length of text; stops early when chunk count reaches max_chunks.
    """
    if not text or not text.strip():
        return

    start = 0
    chunk_index = 0
    text_len = len(text)

    while start < text_len and chunk_index < max_chunks:
        end = min(start + _CHUNK_SIZE_CHARS, text_len)

        if end < text_len:
            boundary = text.rfind(" ", start, end + 100)
            if boundary > start:
                end = boundary

        chunk_text = text[start:end].strip()
        if chunk_text:
            yield FileChunk(
                file_path=path,
                chunk_index=chunk_index,
                content=chunk_text,
                start_char=start,
                end_char=end,
                file_type=file_type,
            )
            chunk_index += 1

        start = max(start + 1, end - _OVERLAP_SIZE_CHARS)

    if chunk_index == 0:
        yield FileChunk(
            file_path=path,
            chunk_index=0,
            content=text.strip(),
            start_char=0,
            end_char=min(len(text), _CHUNK_SIZE_CHARS),
            file_type=file_type,
        )
    elif chunk_index >= max_chunks and start < text_len:
        logger.debug(
            "Chunk cap applied file=%s original_approx_chunks=%d kept_chunks=%d",
            path,
            (text_len // _CHUNK_SIZE_CHARS) + 1,
            chunk_index,
        )


def _read_text_capped(path: Path, max_chars: int = MAX_CHARS_PER_FILE) -> str:
    """
    Read at most max_chars characters from a text file.
    Prevents reading huge logs or dumps into memory.
    """
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            text = f.read(max_chars)
    except OSError as exc:
        raise ParseError(f"Cannot read {path}: {exc}") from exc
    if len(text) >= max_chars:
        logger.debug(
            "File read capped file=%s max_chars=%d truncated=True",
            path,
            max_chars,
        )
    return text


# ---------------------------------------------------------------------------
# Text parser
# ---------------------------------------------------------------------------

_TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst", ".log", ".csv",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".xml", ".html", ".htm", ".tex", ".org",
}


class PlainTextParser(FileParser):
    """Parser for plain-text documents."""

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in _TEXT_EXTENSIONS

    def file_type(self) -> FileType:
        return FileType.TEXT

    def parse(self, path: Path) -> Iterator[FileChunk]:
        if path.stat().st_size > _MAX_FILE_BYTES:
            logger.warning("File too large to index (>50 MB): %s", path)
            return

        text = _read_text_capped(path)
        yield from _chunk_text(text, str(path), self.file_type())


# ---------------------------------------------------------------------------
# Code parser
# ---------------------------------------------------------------------------

_CODE_EXTENSIONS = {
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx",
    ".java", ".kt", ".scala", ".groovy",
    ".c", ".cpp", ".cc", ".cxx", ".h", ".hpp",
    ".go", ".rs", ".swift",
    ".rb", ".php", ".lua", ".r",
    ".sh", ".bash", ".zsh", ".fish",
    ".sql", ".graphql", ".proto",
    ".tf", ".hcl",
    ".dart", ".cs", ".vb",
    ".asm", ".s",
    "Makefile", "Dockerfile",
}


class CodeParser(FileParser):
    """Parser for source code files."""

    def supports(self, path: Path) -> bool:
        return (
            path.suffix.lower() in _CODE_EXTENSIONS
            or path.name in _CODE_EXTENSIONS
        )

    def file_type(self) -> FileType:
        return FileType.CODE

    def parse(self, path: Path) -> Iterator[FileChunk]:
        if path.stat().st_size > _MAX_FILE_BYTES:
            logger.warning("Code file too large (>50 MB): %s", path)
            return

        text = _read_text_capped(path)
        yield from _chunk_text(text, str(path), self.file_type())


# ---------------------------------------------------------------------------
# PDF parser
# ---------------------------------------------------------------------------

def _ocr_available() -> bool:
    """Return True if both pdf2image and pytesseract (+ tesseract binary) are present."""
    try:
        import pdf2image  # noqa: F401, PLC0415
        import pytesseract  # noqa: PLC0415
        pytesseract.get_tesseract_version()  # raises if tesseract binary is missing
        return True
    except Exception:
        return False


def _ocr_page(page_image: object) -> str:
    """
    Run Tesseract OCR on a single PIL image and return the extracted text.

    DPI is already handled by pdf2image (300 dpi default gives good accuracy).
    """
    import pytesseract  # noqa: PLC0415

    return pytesseract.image_to_string(page_image, lang="eng") or ""


class PDFParser(FileParser):
    """
    Parser for PDF documents.

    Strategy (applied per page):
      1. Extract embedded text via pypdf (fast, zero dependencies).
      2. If a page yields no text (scanned / image-only), fall back to OCR
         using pdf2image + pytesseract — provided both are installed and the
         tesseract binary is available.  OCR is skipped gracefully when the
         dependencies are absent; a warning is logged instead.

    Graceful degradation:
      - pypdf missing   → PDF indexing fully disabled.
      - OCR deps missing → scanned pages are silently skipped (text pages still work).
    """

    def __init__(self) -> None:
        self._pypdf_available: bool | None = None
        self._ocr_available: bool | None = None  # lazily checked once

    def supports(self, path: Path) -> bool:
        if path.suffix.lower() != ".pdf":
            return False
        if self._pypdf_available is None:
            try:
                import pypdf  # noqa: F401, PLC0415
                self._pypdf_available = True
            except ImportError:
                self._pypdf_available = False
                logger.warning("pypdf not installed; PDF indexing disabled.")
        return self._pypdf_available

    def file_type(self) -> FileType:
        return FileType.DOCUMENT

    def parse(self, path: Path) -> Iterator[FileChunk]:
        try:
            import pypdf  # noqa: PLC0415
        except ImportError as exc:
            raise ParseError("pypdf is required to parse PDF files.") from exc

        # Lazy-check OCR availability once per parser instance
        if self._ocr_available is None:
            self._ocr_available = _ocr_available()
            if not self._ocr_available:
                logger.debug(
                    "OCR unavailable (pdf2image/pytesseract/tesseract not found); "
                    "scanned PDF pages will be skipped."
                )

        try:
            reader = pypdf.PdfReader(str(path))
            pages_text: list[str] = []
            ocr_page_indices: list[int] = []  # pages that need OCR

            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages_text.append(text)
                else:
                    # Blank page from pypdf — mark for OCR if available
                    pages_text.append("")
                    ocr_page_indices.append(i)

        except Exception as exc:
            raise ParseError(f"Failed to read PDF {path}: {exc}") from exc

        # OCR fallback: render blank pages to images and run Tesseract
        if ocr_page_indices and self._ocr_available:
            logger.debug(
                "PDF has %d image-only page(s), running OCR: %s",
                len(ocr_page_indices),
                path,
            )
            try:
                from pdf2image import convert_from_path  # noqa: PLC0415

                # Render only the pages that need OCR (1-indexed for pdf2image)
                for page_idx in ocr_page_indices:
                    images = convert_from_path(
                        str(path),
                        dpi=300,
                        first_page=page_idx + 1,
                        last_page=page_idx + 1,
                    )
                    if images:
                        ocr_text = _ocr_page(images[0])
                        if ocr_text.strip():
                            pages_text[page_idx] = ocr_text
                            logger.debug(
                                "OCR extracted %d chars from page %d of %s",
                                len(ocr_text),
                                page_idx + 1,
                                path,
                            )
            except Exception as exc:
                # OCR failure must not prevent the text-layer pages from indexing
                logger.warning("OCR failed for %s: %s", path, exc)

        elif ocr_page_indices and not self._ocr_available:
            logger.warning(
                "PDF '%s' has %d image-only page(s) that cannot be indexed "
                "without OCR. Install OCR support: pip install seekr[ocr]",
                path.name,
                len(ocr_page_indices),
            )

        full_text = "\n\n".join(pages_text)
        if len(full_text) > MAX_CHARS_PER_FILE:
            logger.debug(
                "PDF text capped path=%s max_chars=%d truncated=True",
                path,
                MAX_CHARS_PER_FILE,
            )
        full_text = full_text[:MAX_CHARS_PER_FILE]

        if not full_text.strip():
            logger.warning(
                "PDF produced no indexable text (all pages blank after OCR): %s", path
            )
            return

        yield from _chunk_text(full_text, str(path), self.file_type())


# ---------------------------------------------------------------------------
# Image parser
# ---------------------------------------------------------------------------

_IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif",
}


class ImageParser(FileParser):
    """
    Parser for image files. Yields a single placeholder chunk;
    IndexService uses CLIP embedder for images.
    """

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in _IMAGE_EXTENSIONS

    def file_type(self) -> FileType:
        return FileType.IMAGE

    def parse(self, path: Path) -> Iterator[FileChunk]:
        try:
            from PIL import Image  # noqa: PLC0415
            with Image.open(path) as img:
                width, height = img.size
                mode = img.mode
        except ImportError:
            width = height = 0
            mode = "unknown"
        except Exception as exc:
            raise ParseError(f"Cannot open image {path}: {exc}") from exc

        yield FileChunk(
            file_path=str(path),
            chunk_index=0,
            content=f"Image file: {path.name} ({width}x{height} {mode})",
            start_char=0,
            end_char=0,
            file_type=FileType.IMAGE,
        )
