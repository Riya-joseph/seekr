"""
Deprecated shim — imports here will raise a DeprecationWarning in a future release.

Use ``seekr.config.settings`` directly:

    from seekr.config.settings import MAX_CHARS_PER_FILE, CHUNK_SIZE, ...
"""

import warnings

warnings.warn(
    "Importing from seekr.config is deprecated. " "Use seekr.config.settings instead.",
    DeprecationWarning,
    stacklevel=2,
)

from seekr.config.settings import (  # noqa: F401, E402
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
