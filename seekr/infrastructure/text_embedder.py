"""
SentenceTransformerEmbedder — text embedding via sentence-transformers.

Default model: BAAI/bge-small-en-v1.5  (384-dim, ~120 MB, fast on CPU)
Fallback:      sentence-transformers/all-MiniLM-L6-v2 (384-dim, ~80 MB)

Architecture note: This is an "adapter" in ports-and-adapters.  It
implements the EmbeddingModel interface from the domain layer.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

from seekr.domain.exceptions import ModelError
from seekr.domain.interfaces import EmbeddingModel

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
_BATCH_SIZE = 32


class SentenceTransformerEmbedder(EmbeddingModel):
    """
    Text embedder backed by sentence-transformers.

    Lazy-loads the model on first use to keep import time fast.
    Thread-safe for read operations (inference is stateless).
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: str = "cpu",
        cache_dir: Optional[Path] = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._cache_dir = str(Path(cache_dir).expanduser().resolve()) if cache_dir else None
        self._model: Optional[object] = None  # loaded lazily

    # ------------------------------------------------------------------
    # EmbeddingModel interface
    # ------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        return 384  # true for both bge-small and MiniLM-L6

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_text(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of strings using sentence-transformers.

        Normalises vectors to unit length so cosine similarity equals
        dot product (required for FAISS IndexFlatIP).
        """
        if not texts:
            return []

        model = self._get_model()
        try:
            embeddings = model.encode(
                texts,
                batch_size=_BATCH_SIZE,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return embeddings.tolist()
        except Exception as exc:
            raise ModelError(f"Text embedding failed: {exc}") from exc

    def embed_image(self, image_paths: list[Path]) -> list[list[float]]:
        """Not supported — use CLIPEmbedder for images."""
        raise ModelError(
            "SentenceTransformerEmbedder does not support image embedding. "
            "Use CLIPEmbedder instead."
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_model(self) -> object:
        if self._model is None:
            logger.debug("Loading text embedding model: %s", self._model_name)
            try:
                # Pin Hugging Face cache to our data dir so blobs and symlinks stay local
                if self._cache_dir:
                    os.environ["HUGGINGFACE_HUB_CACHE"] = self._cache_dir
                os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
                _tqdm_disable = os.environ.get("TQDM_DISABLE")
                os.environ["TQDM_DISABLE"] = "1"
                try:
                    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

                    # Suppress load-report and progress noise from transformers/HF
                    for _noisy in ("huggingface_hub", "transformers", "transformers.modeling_utils"):
                        logging.getLogger(_noisy).setLevel(logging.ERROR)

                    self._model = SentenceTransformer(
                        self._model_name,
                        device=self._device,
                        cache_folder=self._cache_dir,
                    )
                    logger.debug("Text model loaded (dim=%d)", self.dimension)
                finally:
                    if _tqdm_disable is None:
                        os.environ.pop("TQDM_DISABLE", None)
                    else:
                        os.environ["TQDM_DISABLE"] = _tqdm_disable
            except ImportError as exc:
                raise ModelError(
                    "sentence-transformers is not installed. "
                    "Run: pip install sentence-transformers"
                ) from exc
            except Exception as exc:
                raise ModelError(f"Failed to load model '{self._model_name}': {exc}") from exc
        return self._model
