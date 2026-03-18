"""
Centralized logging configuration for Seekr.

Import and call configure_logging() once at process startup (done in cli/main.py).
All other modules obtain their logger via logging.getLogger(__name__) without
configuring handlers — this module owns the handler setup.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def configure_logging(
    verbose: bool = False,
    log_file: Optional[Path] = None,
) -> None:
    """
    Configure the root "seekr" logger.

    Args:
        verbose:  When True, set level to DEBUG; otherwise WARNING for the
                  root logger and INFO for the seekr namespace.
        log_file: Optional path to a file that receives all log output.
                  Used in daemon (background) mode where stdout is the log file.
    """
    seekr_logger = logging.getLogger("seekr")
    seekr_logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Avoid adding duplicate handlers if called more than once
    if seekr_logger.handlers:
        return

    formatter = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    if log_file is not None:
        # Daemon mode: write structured text to a file instead of the terminal
        file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
        file_handler.setFormatter(formatter)
        seekr_logger.addHandler(file_handler)
    else:
        # Interactive mode: use Rich for pretty terminal output when available
        try:
            from rich.logging import RichHandler  # noqa: PLC0415

            rich_handler = RichHandler(
                rich_tracebacks=True,
                show_path=False,
                markup=False,
            )
            seekr_logger.addHandler(rich_handler)
        except ImportError:
            stream_handler = logging.StreamHandler(sys.stderr)
            stream_handler.setFormatter(formatter)
            seekr_logger.addHandler(stream_handler)

    # Silence noisy third-party loggers at WARNING or higher only.
    # transformers.modeling_utils emits "BertModel LOAD REPORT" at INFO — suppress it.
    for noisy in (
        "huggingface_hub",
        "transformers",
        "transformers.modeling_utils",
        "transformers.configuration_utils",
        "transformers.tokenization_utils_base",
        "sentence_transformers",
        "PIL",
        "filelock",
        "urllib3",
    ):
        logging.getLogger(noisy).setLevel(logging.ERROR)
