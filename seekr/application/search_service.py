"""
SearchService — application-layer orchestrator for semantic search.

Accepts a natural-language query string (or a path to an image for
image-to-image search), embeds it, queries the VectorStore, and
resolves the raw (chunk_id, score) pairs into rich SearchResult objects
by looking up metadata in the MetadataStore.

No FAISS, no SentenceTransformer — only domain interfaces.
"""

from __future__ import annotations

import logging
from pathlib import Path

from seekr.domain.entities import FileType, SearchResult
from seekr.domain.exceptions import SearchError
from seekr.domain.interfaces import EmbeddingModel, MetadataStore, VectorStore

logger = logging.getLogger(__name__)

_SNIPPET_CHARS = 280
# Reciprocal Rank Fusion constant (k=60 is standard for multi-list fusion)
_RRF_K = 60
# When filtering by path, fetch extra candidates so enough pass the filter
_PATH_FILTER_FETCH_MULTIPLIER = 6


class SearchService:
    """
    Application service for querying the semantic index.

    The same embedding space is used for both text and images (CLIP),
    so a text query can match images and vice-versa.
    """

    def __init__(
        self,
        text_embedder: EmbeddingModel,
        image_embedder: EmbeddingModel,
        vector_store: VectorStore,
        metadata_store: MetadataStore,
        image_vector_store: VectorStore | None = None,
    ) -> None:
        self._text_embedder = text_embedder
        self._image_embedder = image_embedder
        self._vector_store = vector_store
        self._image_vector_store = image_vector_store
        self._metadata_store = metadata_store

    def search(
        self,
        query: str,
        top_k: int = 10,
        file_type_filter: FileType | None = None,
        path_prefix: Path | None = None,
    ) -> list[SearchResult]:
        """
        Search the index with a natural-language query.

        When file_type_filter is None, both the text and image vector stores are
        searched; results are merged using Reciprocal Rank Fusion (RRF) so both
        lists contribute fairly. When file_type_filter is IMAGE, only the image
        (CLIP) store is searched.

        Args:
            query:            Free-text search query.
            top_k:            Maximum number of results to return.
            file_type_filter: If set, restrict results to this FileType.
            path_prefix:      If set, only return results under this path (file or directory).

        Returns:
            Sorted list of SearchResult objects (best match first).

        Raises:
            SearchError: If embedding or vector lookup fails.
        """
        if not query.strip():
            raise SearchError("Query must not be empty.")

        fetch_k = top_k * _PATH_FILTER_FETCH_MULTIPLIER if path_prefix is not None else top_k

        # Image-only: use CLIP and image vector store
        if file_type_filter == FileType.IMAGE and self._image_vector_store is not None:
            try:
                vectors = self._image_embedder.embed_text([query])
                query_vector = vectors[0]
            except Exception as exc:
                raise SearchError(f"Failed to embed query for image search: {exc}") from exc
            results = self._run_search(
                query_vector, fetch_k, file_type_filter, self._image_vector_store
            )
            return self._apply_path_filter(results, path_prefix, top_k)

        # No type filter: search both text and image stores, merge with RRF
        if file_type_filter is None and self._image_vector_store is not None:
            try:
                text_vectors = self._text_embedder.embed_text([query])
                image_vectors = self._image_embedder.embed_text([query])
            except Exception as exc:
                raise SearchError(f"Failed to embed query: {exc}") from exc
            text_results = self._run_search(text_vectors[0], fetch_k, None, self._vector_store)
            image_results = self._run_search(
                image_vectors[0], fetch_k, None, self._image_vector_store
            )
            results = self._merge_rrf(text_results, image_results, fetch_k)
            return self._apply_path_filter(results, path_prefix, top_k)

        # Text/code/document filter or no image store: text store only
        try:
            vectors = self._text_embedder.embed_text([query])
            query_vector = vectors[0]
        except Exception as exc:
            raise SearchError(f"Failed to embed query: {exc}") from exc

        results = self._run_search(query_vector, fetch_k, file_type_filter, self._vector_store)
        return self._apply_path_filter(results, path_prefix, top_k)

    def search_by_image(
        self,
        image_path: Path,
        top_k: int = 10,
        file_type_filter: FileType | None = None,
        path_prefix: Path | None = None,
    ) -> list[SearchResult]:
        """
        Search the index using an image as the query.

        Args:
            image_path:       Path to a query image.
            top_k:            Maximum number of results to return.
            file_type_filter: Optional filter by FileType.
            path_prefix:      If set, only return results under this path.

        Returns:
            Sorted list of SearchResult objects.
        """
        if not image_path.exists():
            raise SearchError(f"Image file not found: {image_path}")

        try:
            vectors = self._image_embedder.embed_image([image_path])
            query_vector = vectors[0]
        except Exception as exc:
            raise SearchError(f"Failed to embed query image: {exc}") from exc

        fetch_k = top_k * _PATH_FILTER_FETCH_MULTIPLIER if path_prefix is not None else top_k
        store = (
            self._image_vector_store if self._image_vector_store is not None else self._vector_store
        )
        results = self._run_search(query_vector, fetch_k, file_type_filter, store)
        return self._apply_path_filter(results, path_prefix, top_k)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_path_filter(
        self,
        results: list[SearchResult],
        path_prefix: Path | None,
        top_k: int,
    ) -> list[SearchResult]:
        """If path_prefix is set, keep only results under that path; then return first top_k."""
        if path_prefix is None:
            return results[:top_k]
        filtered = [r for r in results if self._is_under_path(r.file_path, path_prefix)]
        return filtered[:top_k]

    @staticmethod
    def _is_under_path(file_path: str, prefix: Path) -> bool:
        """True if file_path is equal to prefix or under it (normalized)."""
        try:
            resolved = Path(file_path).expanduser().resolve()
            base = prefix.expanduser().resolve()
            if resolved == base:
                return True
            # Check if resolved is under base using path parts (avoids slash/OS issues)
            base_parts = base.parts
            file_parts = resolved.parts
            if len(file_parts) < len(base_parts):
                return False
            return file_parts[: len(base_parts)] == base_parts
        except (ValueError, OSError):
            return False

    @staticmethod
    def _merge_rrf(
        text_results: list[SearchResult],
        image_results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """
        Merge text and image results using Reciprocal Rank Fusion (RRF).

        RRF score = 1 / (k + rank) per list. Same file in both lists gets
        combined RRF (sum). Results are deduplicated by file_path, then
        sorted by combined score.
        """
        by_path: dict[str, tuple[float, SearchResult]] = {}
        for rank, result in enumerate(text_results, start=1):
            rrf = 1.0 / (_RRF_K + rank)
            key = result.file_path
            if key in by_path:
                prev_rrf, prev = by_path[key]
                by_path[key] = (prev_rrf + rrf, prev)
            else:
                by_path[key] = (rrf, result)
        for rank, result in enumerate(image_results, start=1):
            rrf = 1.0 / (_RRF_K + rank)
            key = result.file_path
            if key in by_path:
                prev_rrf, prev = by_path[key]
                # Keep the result, add RRF to combined score
                by_path[key] = (prev_rrf + rrf, prev)
            else:
                by_path[key] = (rrf, result)
        combined = sorted(by_path.values(), key=lambda x: x[0], reverse=True)
        return [result for _, result in combined[:top_k]]

    def _run_search(
        self,
        query_vector: list[float],
        top_k: int,
        file_type_filter: FileType | None,
        vector_store: VectorStore | None = None,
    ) -> list[SearchResult]:
        """Execute vector search and resolve results to domain objects."""
        store = vector_store if vector_store is not None else self._vector_store
        # Fetch extra candidates to allow post-filtering
        fetch_k = top_k * 4 if file_type_filter else top_k
        try:
            raw = store.search(query_vector, top_k=fetch_k)
        except Exception as exc:
            raise SearchError(f"Vector store search failed: {exc}") from exc

        results: list[SearchResult] = []
        seen_paths: set[str] = set()  # de-duplicate by file path

        for chunk_id, score in raw:
            file_path, chunk_idx = self._split_chunk_id(chunk_id)

            record = self._metadata_store.get(file_path)
            if record is None:
                logger.debug("Chunk %s has no metadata record — stale?", chunk_id)
                continue

            if file_type_filter and record.file_type != file_type_filter:
                continue

            # For text files, prefer showing unique files unless the query
            # specifically targets chunks.
            dedup_key = file_path
            if dedup_key in seen_paths:
                continue
            seen_paths.add(dedup_key)

            snippet = self._load_snippet(Path(file_path), chunk_idx)

            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    file_path=file_path,
                    chunk_index=chunk_idx,
                    score=score,
                    snippet=snippet,
                    file_type=record.file_type,
                    modified_at=record.modified_at,
                )
            )

            if len(results) >= top_k:
                break

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    @staticmethod
    def _split_chunk_id(chunk_id: str) -> tuple[str, int]:
        """Split 'path::index' into (path, index)."""
        if "::" in chunk_id:
            path, _, idx = chunk_id.rpartition("::")
            return path, int(idx)
        return chunk_id, 0

    @staticmethod
    def _load_snippet(path: Path, chunk_idx: int) -> str:
        """
        Load a short text snippet from a file for display.

        For images, returns a descriptive string.
        For text, reads a fixed number of characters.
        """
        if not path.exists():
            return "[file not found]"

        suffix = path.suffix.lower()
        image_suffixes = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}
        if suffix in image_suffixes:
            return f"[image: {path.name}]"

        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                # Approximate: jump to rough char position and take a window
                offset = chunk_idx * (_SNIPPET_CHARS * 2)
                fh.seek(offset)
                text = fh.read(_SNIPPET_CHARS * 3)
                snippet = text[:_SNIPPET_CHARS].strip()
                if len(text) > _SNIPPET_CHARS:
                    snippet += "…"
                return snippet
        except OSError:
            return "[could not read file]"
