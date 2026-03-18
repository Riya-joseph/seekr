"""
WatchdogFileWatcher — watchdog-backed implementation of the FileWatcher port.

Uses watchdog's inotify backend on Linux for low-overhead kernel-level
file system event monitoring.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Optional

from seekr.domain.exceptions import WatcherError
from seekr.domain.interfaces import FileWatcher

logger = logging.getLogger(__name__)

# Debounce interval in seconds — prevents double-firing on rapid saves
_DEBOUNCE_SECONDS = 1.0


class WatchdogFileWatcher(FileWatcher):
    """
    Concrete FileWatcher using the watchdog library.

    Debounces rapid file events to avoid redundant indexing when an
    editor saves multiple times in quick succession.
    """

    def __init__(self) -> None:
        self._observer: Optional[object] = None
        self._handlers: list[object] = []

    def start(
        self,
        paths: list[Path],
        on_created: Callable[[Path], None],
        on_modified: Callable[[Path], None],
        on_deleted: Callable[[Path], None],
    ) -> None:
        """
        Start watching all given paths recursively.

        Args:
            paths:       Directories (or files) to monitor.
            on_created:  Callback invoked when a file is created.
            on_modified: Callback invoked when a file is modified.
            on_deleted:  Callback invoked when a file is deleted.
        """
        try:
            from watchdog.observers import Observer  # noqa: PLC0415
            from watchdog.events import FileSystemEventHandler  # noqa: PLC0415
        except ImportError as exc:
            raise WatcherError(
                "watchdog is not installed. Run: pip install watchdog"
            ) from exc

        class _Handler(FileSystemEventHandler):
            def __init__(self_handler) -> None:
                super().__init__()
                self_handler._last_event: dict[str, float] = {}

            def _should_process(self_handler, path: str) -> bool:
                now = time.monotonic()
                last = self_handler._last_event.get(path, 0.0)
                if now - last < _DEBOUNCE_SECONDS:
                    return False
                self_handler._last_event[path] = now
                return True

            def on_created(self_handler, event) -> None:
                if event.is_directory:
                    return
                p = Path(event.src_path)
                if self_handler._should_process(str(p)):
                    on_created(p)

            def on_modified(self_handler, event) -> None:
                if event.is_directory:
                    return
                p = Path(event.src_path)
                if self_handler._should_process(str(p)):
                    on_modified(p)

            def on_deleted(self_handler, event) -> None:
                if event.is_directory:
                    return
                p = Path(event.src_path)
                if self_handler._should_process(str(p)):
                    on_deleted(p)

            def on_moved(self_handler, event) -> None:
                if event.is_directory:
                    return
                # Treat move as delete-old + create-new
                src = Path(event.src_path)
                dst = Path(event.dest_path)
                if self_handler._should_process(str(src)):
                    on_deleted(src)
                if self_handler._should_process(str(dst)):
                    on_created(dst)

        self._observer = Observer()
        for path in paths:
            path = path.resolve()
            if not path.exists():
                logger.warning("Watch path does not exist: %s", path)
                continue
            handler = _Handler()
            self._handlers.append(handler)
            self._observer.schedule(handler, str(path), recursive=True)
            logger.info("Watching: %s", path)

        self._observer.start()
        logger.info("Watchdog observer started.")

    def stop(self) -> None:
        """Stop the observer and join its thread."""
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5.0)
            self._observer = None
            self._handlers.clear()
            logger.info("Watchdog observer stopped.")
