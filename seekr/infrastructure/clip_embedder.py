"""
CLIPEmbedder — joint text+image embedding via openai/clip-vit-base-patch32.

CLIP projects both text and images into the same 512-dimensional space,
enabling cross-modal search:
  - "a cat sitting on a sofa" will match photos of cats
  - an image can be used as a query to find similar images or documents

Architecture note: This class implements *both* embed_text and embed_image
so that image-heavy indices can use a single unified model.  The
SentenceTransformerEmbedder is preferred for pure-text indices because
it produces higher-quality text embeddings.

For maximum recall, Seekr maintains **two parallel vector stores**:
  1. A text-only store using SentenceTransformerEmbedder.
  2. A CLIP store for images (and optionally text-in-images via OCR).

The SearchService is responsible for selecting which store to query.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from seekr.domain.exceptions import ModelError
from seekr.domain.interfaces import EmbeddingModel

logger = logging.getLogger(__name__)

_CLIP_MODEL = "openai/clip-vit-base-patch32"
_CLIP_DIM = 512


class CLIPEmbedder(EmbeddingModel):
    """
    Dual-modality embedder using CLIP.

    Supports both text queries (for image retrieval) and image queries
    (for image similarity search).  Uses the transformers / PIL stack
    to avoid the openai-clip dependency.
    """

    def __init__(
        self,
        model_name: str = _CLIP_MODEL,
        device: str = "cpu",
        cache_dir: Path | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._cache_dir = str(Path(cache_dir).expanduser().resolve()) if cache_dir else None
        self._model: Any = None
        self._processor: Any = None

    # ------------------------------------------------------------------
    # EmbeddingModel interface
    # ------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        return _CLIP_DIM

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_text(self, texts: list[str]) -> list[list[float]]:
        """
        Embed text strings into CLIP's joint space.

        Useful for text-to-image queries.
        """
        if not texts:
            return []
        model, processor = self._get_model()
        try:
            import torch

            inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model.get_text_features(**inputs)
                features = self._features_tensor(out)
                features = features / features.norm(dim=-1, keepdim=True)
            return features.cpu().numpy().tolist()  # type: ignore[no-any-return]
        except Exception as exc:
            raise ModelError(f"CLIP text embedding failed: {exc}") from exc

    def embed_image(self, image_paths: list[Path]) -> list[list[float]]:
        """
        Embed images into CLIP's joint space.

        Args:
            image_paths: Paths to image files (JPEG, PNG, …).

        Returns:
            List of 512-dim unit-norm vectors.
        """
        if not image_paths:
            return []
        model, processor = self._get_model()
        try:
            import torch
            from PIL import Image

            images = [Image.open(p).convert("RGB") for p in image_paths]
            inputs = processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model.get_image_features(**inputs)
                features = self._features_tensor(out)
                features = features / features.norm(dim=-1, keepdim=True)
            return features.cpu().numpy().tolist()  # type: ignore[no-any-return]
        except Exception as exc:
            raise ModelError(f"CLIP image embedding failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _features_tensor(out: Any) -> Any:
        """Get embedding tensor from CLIP output (handles tensor or BaseModelOutputWithPooling)."""
        import torch

        if isinstance(out, torch.Tensor):
            return out
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state
        raise ModelError(f"Unexpected CLIP output type: {type(out)}")

    def _get_model(self) -> tuple[Any, Any]:
        if self._model is None:
            logger.debug("Loading CLIP model: %s", self._model_name)
            # Check all required packages upfront so the error message is accurate.
            missing = []
            try:
                import torch  # noqa: F401
            except ImportError:
                missing.append("torch")
            try:
                import transformers  # noqa: F401
            except ImportError:
                missing.append("transformers")
            try:
                import PIL  # noqa: F401
            except ImportError:
                missing.append("Pillow")
            if missing:
                raise ModelError(
                    f"Missing packages required for image indexing/search: {', '.join(missing)}. "
                    f"Run: pip install {' '.join(missing)}"
                )

            try:
                # Pin Hugging Face cache to our data dir so blobs and symlinks stay local
                if self._cache_dir:
                    os.environ["HUGGINGFACE_HUB_CACHE"] = self._cache_dir
                os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
                _tqdm_disable = os.environ.get("TQDM_DISABLE")
                os.environ["TQDM_DISABLE"] = "1"
                try:
                    import transformers as _tf
                    from transformers import CLIPModel, CLIPProcessor

                    _tf.logging.set_verbosity_error()  # type: ignore[no-untyped-call]
                    if hasattr(_tf.utils.logging, "disable_progress_bar"):
                        _tf.utils.logging.disable_progress_bar()  # type: ignore[no-untyped-call]
                    else:
                        _tf.logging.disable_progress_bar()  # type: ignore[no-untyped-call]
                    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

                    self._processor = CLIPProcessor.from_pretrained(
                        self._model_name, cache_dir=self._cache_dir
                    )
                    _clip = CLIPModel.from_pretrained(self._model_name, cache_dir=self._cache_dir)
                    self._model = _clip.to(self._device)  # type: ignore[arg-type]
                    self._model.eval()
                    logger.debug("CLIP model loaded (dim=%d, device=%s)", _CLIP_DIM, self._device)
                finally:
                    if _tqdm_disable is None:
                        os.environ.pop("TQDM_DISABLE", None)
                    else:
                        os.environ["TQDM_DISABLE"] = _tqdm_disable
            except ModelError:
                raise
            except Exception as exc:
                raise ModelError(f"Failed to load CLIP model: {exc}") from exc
        return self._model, self._processor
