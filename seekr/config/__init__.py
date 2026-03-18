"""Configuration package — imports all settings for convenience."""

from seekr.config.settings import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CLIP_DIM,
    CLIP_MODEL_NAME,
    DEFAULT_DEVICE,
    MAX_CHARS_PER_FILE,
    MAX_CHUNKS_PER_FILE,
    NUM_INDEX_WORKERS,
    TEXT_DIM,
    TEXT_MODEL_NAME,
)

__all__ = [
    "CHUNK_OVERLAP",
    "CHUNK_SIZE",
    "CLIP_DIM",
    "CLIP_MODEL_NAME",
    "DEFAULT_DEVICE",
    "MAX_CHARS_PER_FILE",
    "MAX_CHUNKS_PER_FILE",
    "NUM_INDEX_WORKERS",
    "TEXT_DIM",
    "TEXT_MODEL_NAME",
]
