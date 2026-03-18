"""Configuration package — imports all settings for convenience."""

from seekr.config.settings import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    MAX_CHARS_PER_FILE,
    MAX_CHUNKS_PER_FILE,
    NUM_INDEX_WORKERS,
    TEXT_MODEL_NAME,
    CLIP_MODEL_NAME,
    TEXT_DIM,
    CLIP_DIM,
    DEFAULT_DEVICE,
)

__all__ = [
    "CHUNK_OVERLAP",
    "CHUNK_SIZE",
    "MAX_CHARS_PER_FILE",
    "MAX_CHUNKS_PER_FILE",
    "NUM_INDEX_WORKERS",
    "TEXT_MODEL_NAME",
    "CLIP_MODEL_NAME",
    "TEXT_DIM",
    "CLIP_DIM",
    "DEFAULT_DEVICE",
]
