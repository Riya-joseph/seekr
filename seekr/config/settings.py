"""
Centralized configuration for Seekr.

All tunable constants live here. Each setting can be overridden via environment
variables so the same binary works in development, CI, and production without
code changes.

Environment variable mapping:
  SEEKR_MAX_CHARS_PER_FILE   → MAX_CHARS_PER_FILE
  SEEKR_MAX_CHUNKS_PER_FILE  → MAX_CHUNKS_PER_FILE
  SEEKR_CHUNK_SIZE           → CHUNK_SIZE
  SEEKR_CHUNK_OVERLAP        → CHUNK_OVERLAP
  SEEKR_NUM_INDEX_WORKERS    → NUM_INDEX_WORKERS
  SEEKR_TEXT_MODEL           → TEXT_MODEL_NAME
  SEEKR_CLIP_MODEL           → CLIP_MODEL_NAME
  SEEKR_DEVICE               → DEFAULT_DEVICE
"""

from __future__ import annotations

import os


def _int_env(name: str, default: int) -> int:
    """Read an integer environment variable, falling back to *default*."""
    raw = os.environ.get(name, "")
    if raw.strip():
        try:
            return int(raw)
        except ValueError:
            pass
    return default


def _str_env(name: str, default: str) -> str:
    """Read a string environment variable, falling back to *default*."""
    return os.environ.get(name, default).strip() or default


# ---------------------------------------------------------------------------
# Chunking / indexing limits
# ---------------------------------------------------------------------------

# Maximum characters read from a single file. Files larger than this are
# truncated at read time, preventing huge logs from bloating memory and index.
MAX_CHARS_PER_FILE: int = _int_env("SEEKR_MAX_CHARS_PER_FILE", 1_000_000)

# Hard cap on chunks produced per file. Stops chunk generation so that a
# single enormous file cannot dominate the index with thousands of vectors.
MAX_CHUNKS_PER_FILE: int = _int_env("SEEKR_MAX_CHUNKS_PER_FILE", 200)

# Chunk size in token-equivalents (~4 chars per token for Latin text).
CHUNK_SIZE: int = _int_env("SEEKR_CHUNK_SIZE", 500)

# Overlap between consecutive chunks (token-equivalents).
# Overlap preserves context around chunk boundaries for better retrieval.
CHUNK_OVERLAP: int = _int_env("SEEKR_CHUNK_OVERLAP", 50)

# ---------------------------------------------------------------------------
# Worker pool
# ---------------------------------------------------------------------------

# Number of parallel background indexing threads.
NUM_INDEX_WORKERS: int = _int_env("SEEKR_NUM_INDEX_WORKERS", 4)

# ---------------------------------------------------------------------------
# Embedding models
# ---------------------------------------------------------------------------

# Default text embedding model (384-dim). Produces high-quality text vectors.
TEXT_MODEL_NAME: str = _str_env("SEEKR_TEXT_MODEL", "BAAI/bge-small-en-v1.5")

# CLIP model for joint text+image embeddings (512-dim).
CLIP_MODEL_NAME: str = _str_env("SEEKR_CLIP_MODEL", "openai/clip-vit-base-patch32")

# Embedding dimensions (must match model output).
TEXT_DIM: int = 384
CLIP_DIM: int = 512

# Compute device: "cpu", "cuda", "mps".
DEFAULT_DEVICE: str = _str_env("SEEKR_DEVICE", "cpu")
