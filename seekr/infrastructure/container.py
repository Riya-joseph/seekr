"""
Container — dependency injection factory for Seekr.

Assembles the full object graph from infrastructure implementations,
following the dependency rule: outer layers depend on inner layers only.

  CLI → Application → Domain ← Infrastructure

The Container is the ONLY place where infrastructure types are imported
together.  Application services and CLI commands never import from
infrastructure directly.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path

from seekr.application.index_service import IndexService
from seekr.application.search_service import SearchService
from seekr.application.watcher_service import WatcherService
from seekr.config.settings import (
    CLIP_DIM,
    CLIP_MODEL_NAME,
    DEFAULT_DEVICE,
    TEXT_DIM,
    TEXT_MODEL_NAME,
)
from seekr.domain.interfaces import (
    EmbeddingModel,
    FileParser,
    IndexQueue,
    MetadataStore,
    VectorStore,
)

logger = logging.getLogger(__name__)

_DEFAULT_DATA_DIR: Path = Path.home() / ".seekr"


# Embedding model dimension constants (kept as module-level aliases for clarity)
_TEXT_DIM: int = TEXT_DIM
_CLIP_DIM: int = CLIP_DIM


class Container:
    """
    Wires up all services with concrete infrastructure implementations.

    Designed for single-process use.  For a server version, split into
    separate container classes per service boundary.
    """

    def __init__(
        self,
        data_dir: Path = _DEFAULT_DATA_DIR,
        text_model: str = TEXT_MODEL_NAME,
        clip_model: str = CLIP_MODEL_NAME,
        device: str = DEFAULT_DEVICE,
    ) -> None:
        self._data_dir = Path(data_dir).expanduser().resolve()
        self._text_model_name = text_model
        self._clip_model_name = clip_model
        self._device = device

        self._data_dir.mkdir(parents=True, exist_ok=True)

        # Singletons — lazy-initialised
        self._text_embedder: EmbeddingModel | None = None
        self._clip_embedder: EmbeddingModel | None = None
        self._text_vector_store: VectorStore | None = None
        self._clip_vector_store: VectorStore | None = None
        self._metadata_store: MetadataStore | None = None
        self._index_queue: IndexQueue | None = None

    # ------------------------------------------------------------------
    # Infrastructure singletons
    # ------------------------------------------------------------------

    @property
    def text_embedder(self) -> EmbeddingModel:
        if self._text_embedder is None:
            from seekr.infrastructure.text_embedder import SentenceTransformerEmbedder

            self._text_embedder = SentenceTransformerEmbedder(
                model_name=self._text_model_name,
                device=self._device,
                cache_dir=self._data_dir / "models",
            )
        return self._text_embedder

    @property
    def clip_embedder(self) -> EmbeddingModel:
        if self._clip_embedder is None:
            from seekr.infrastructure.clip_embedder import CLIPEmbedder

            self._clip_embedder = CLIPEmbedder(
                model_name=self._clip_model_name,
                device=self._device,
                cache_dir=self._data_dir / "models",
            )
        return self._clip_embedder

    @property
    def text_vector_store(self) -> VectorStore:
        if self._text_vector_store is None:
            from seekr.infrastructure.faiss_store import FAISSVectorStore

            store = FAISSVectorStore(
                store_dir=self._data_dir / "text_index",
                dimension=_TEXT_DIM,
            )
            store.load()
            self._text_vector_store = store
        return self._text_vector_store

    @property
    def clip_vector_store(self) -> VectorStore:
        if self._clip_vector_store is None:
            from seekr.infrastructure.faiss_store import FAISSVectorStore

            store = FAISSVectorStore(
                store_dir=self._data_dir / "clip_index",
                dimension=_CLIP_DIM,
            )
            store.load()
            self._clip_vector_store = store
        return self._clip_vector_store

    @property
    def metadata_store(self) -> MetadataStore:
        if self._metadata_store is None:
            from seekr.infrastructure.sqlite_store import SQLiteMetadataStore

            self._metadata_store = SQLiteMetadataStore(db_path=self._data_dir / "metadata.db")
        return self._metadata_store

    @property
    def index_queue(self) -> IndexQueue:
        if self._index_queue is None:
            from seekr.infrastructure.queue.index_queue import SQLiteIndexQueue

            self._index_queue = SQLiteIndexQueue(db_path=self._data_dir / "metadata.db")
        return self._index_queue

    # ------------------------------------------------------------------
    # Application services
    # ------------------------------------------------------------------

    def make_parsers(self) -> list[FileParser]:
        """Return all supported file parsers in priority order."""
        from seekr.infrastructure.parsers import (
            CodeParser,
            ImageParser,
            PDFParser,
            PlainTextParser,
        )

        return [
            ImageParser(),
            PDFParser(),
            CodeParser(),
            PlainTextParser(),
        ]

    def dry_run_service(self) -> IndexService:
        """
        Minimal IndexService for --dry-run only.

        Dry-run only walks the file tree and estimates chunk counts — it never
        calls embed_text, embed_image, or touches any vector store.  Building
        the full index_service() would trigger lazy model loading (hundreds of
        MB downloaded/loaded from disk), causing an apparent hang.

        This method creates an IndexService backed by no-op stubs for every
        dependency that dry_run() never touches, so it returns immediately.
        """
        from seekr.domain.interfaces import EmbeddingModel, VectorStore

        class _NoOpEmbedder(EmbeddingModel):
            @property
            def dimension(self) -> int:
                return 0

            @property
            def model_name(self) -> str:
                return "dry-run-stub"

            def embed_text(self, texts: list[str]) -> list[list[float]]:
                raise RuntimeError("embed_text must not be called during dry-run")

            def embed_image(self, image_paths: list[Path]) -> list[list[float]]:
                raise RuntimeError("embed_image must not be called during dry-run")

        class _NoOpVectorStore(VectorStore):
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

        stub_embedder = _NoOpEmbedder()
        stub_store = _NoOpVectorStore()

        return IndexService(
            parsers=self.make_parsers(),
            text_embedder=stub_embedder,
            image_embedder=stub_embedder,
            vector_store=stub_store,
            metadata_store=self.metadata_store,
        )

    def index_service(
        self,
        progress_callback: Callable[[str, int, int], None] | None = None,
        use_clip_for_text: bool = False,
        background: bool = False,
    ) -> IndexService:
        """
        Build an IndexService.

        Args:
            progress_callback: Optional fn(file, done, total) for progress display.
            use_clip_for_text: If True, use CLIP embeddings for text files too
                               (enables cross-modal search from text→image).
                               Defaults to False (SentenceTransformer is better
                               for pure text retrieval).
            background: If True, inject the index queue so index_path/index_file
                        only enqueue tasks; workers do the actual indexing.
        """
        text_emb = self.clip_embedder if use_clip_for_text else self.text_embedder
        text_store = self.clip_vector_store if use_clip_for_text else self.text_vector_store
        # When using separate text store, images go to CLIP store (512-dim); when use_clip_for_text, both use same store.
        image_store = None if use_clip_for_text else self.clip_vector_store
        queue = self.index_queue if background else None

        return IndexService(
            parsers=self.make_parsers(),
            text_embedder=text_emb,
            image_embedder=self.clip_embedder,
            vector_store=text_store,
            metadata_store=self.metadata_store,
            progress_callback=progress_callback,
            image_vector_store=image_store,
            queue=queue,
        )

    def start_index_workers(
        self,
        num_workers: int = 4,
        daemon: bool = True,
        stop_event: threading.Event | None = None,
        ignore_patterns: set[str] | None = None,
    ) -> None:
        """
        Start background worker threads.

        Args:
            num_workers:     Number of parallel threads.
            daemon:          If True, threads don't block process exit.
            stop_event:      Pass a threading.Event for watch mode — workers poll
                             until stop_event.set() is called. If None, workers exit
                             when the queue is empty (drain mode for seekr index).
            ignore_patterns: Path-component patterns passed to every worker so
                             files from excluded directories are never indexed,
                             even if they are already sitting in the queue from
                             a previous run.
        """
        from seekr.infrastructure.workers.index_worker import run_worker_pool

        queue = self.index_queue
        index_svc = self.index_service(background=False)
        run_worker_pool(
            queue,
            index_svc,
            num_workers=num_workers,
            daemon=daemon,
            stop_event=stop_event,
            ignore_patterns=ignore_patterns,
        )

    def search_service(self) -> SearchService:
        """Build a SearchService using both text and CLIP stores."""
        return SearchService(
            text_embedder=self.text_embedder,
            image_embedder=self.clip_embedder,
            vector_store=self.text_vector_store,
            metadata_store=self.metadata_store,
            image_vector_store=self.clip_vector_store,
        )

    def watcher_service(
        self,
        on_event: Callable[[str, Path], None] | None = None,
        use_queue: bool = False,
        ignore_patterns: set[str] | None = None,
    ) -> WatcherService:
        """
        Build a WatcherService.

        If use_queue is True, index_service will enqueue files for background workers
        instead of indexing synchronously. Start workers via start_index_workers() when
        using this (e.g. for seekr watch).
        """
        from seekr.infrastructure.watcher import WatchdogFileWatcher

        return WatcherService(
            file_watcher=WatchdogFileWatcher(),
            index_service=self.index_service(background=use_queue),
            on_event=on_event,
            ignore_patterns=ignore_patterns,
        )

    @property
    def data_dir(self) -> Path:
        return self._data_dir
