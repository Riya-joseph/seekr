"""
WatcherService — application-layer orchestrator for continuous indexing.

Wraps a FileWatcher implementation and delegates file-system events to
IndexService.  All infrastructure details (watchdog, inotify) are hidden
behind the FileWatcher interface.
"""

from __future__ import annotations

import logging
import signal
import time
from collections.abc import Callable
from pathlib import Path

from seekr.application.index_service import IndexService
from seekr.domain.exceptions import WatcherError
from seekr.domain.interfaces import FileWatcher

logger = logging.getLogger(__name__)


class WatcherService:
    """
    Application service that keeps the index up-to-date in real time.

    Usage::

        svc = WatcherService(watcher, index_service)
        svc.start([Path("/home/user/documents")])
        # blocks until SIGINT / SIGTERM
        svc.stop()
    """

    def __init__(
        self,
        file_watcher: FileWatcher,
        index_service: IndexService,
        on_event: Callable[[str, Path], None] | None = None,
        ignore_patterns: set[str] | None = None,
    ) -> None:
        """
        Args:
            file_watcher:     Infrastructure FileWatcher implementation.
            index_service:    Application IndexService for (re-)indexing.
            on_event:         Optional callback(event_type, path) for UIs.
            ignore_patterns:  Set of patterns to skip (from ignore.py).
        """
        self._watcher = file_watcher
        self._index_service = index_service
        self._on_event = on_event
        self._ignore_patterns: set[str] = ignore_patterns or set()
        self._running = False

    def start(self, paths: list[Path], blocking: bool = True) -> None:
        """
        Begin watching the given paths.

        Args:
            paths:    Directories (or files) to watch recursively.
            blocking: If True, block until the process receives SIGINT/SIGTERM.
        """
        if self._running:
            raise WatcherError("Watcher is already running.")

        logger.info("Starting file watcher on: %s", paths)
        self._running = True

        self._watcher.start(
            paths=paths,
            on_created=self._handle_created,
            on_modified=self._handle_modified,
            on_deleted=self._handle_deleted,
        )

        if blocking:
            self._block_until_signal()

    def stop(self) -> None:
        """Stop the file watcher."""
        if not self._running:
            return
        self._running = False
        self._watcher.stop()
        logger.info("File watcher stopped.")

    # ------------------------------------------------------------------
    # Event handlers (called from watchdog thread)
    # ------------------------------------------------------------------

    def _path_ignored(self, path: Path) -> bool:
        if not self._ignore_patterns:
            return False
        from seekr.domain.patterns import is_ignored

        return is_ignored(path, self._ignore_patterns)

    def _handle_created(self, path: Path) -> None:
        if self._path_ignored(path):
            logger.debug("Ignoring created event (excluded): %s", path)
            return
        logger.info("File created: %s", path)
        self._emit("created", path)
        try:
            self._index_service.index_file(path)
        except Exception as exc:
            logger.error("Failed to index created file %s: %s", path, exc)

    def _handle_modified(self, path: Path) -> None:
        if self._path_ignored(path):
            logger.debug("Ignoring modified event (excluded): %s", path)
            return
        logger.info("File modified: %s", path)
        self._emit("modified", path)
        try:
            self._index_service.index_file(path)
        except Exception as exc:
            logger.error("Failed to re-index modified file %s: %s", path, exc)

    def _handle_deleted(self, path: Path) -> None:
        if self._path_ignored(path):
            logger.debug("Ignoring deleted event (excluded): %s", path)
            return
        logger.info("File deleted: %s", path)
        self._emit("deleted", path)
        try:
            self._index_service.remove_file(path)
        except Exception as exc:
            logger.error("Failed to remove deleted file %s: %s", path, exc)

    def _emit(self, event_type: str, path: Path) -> None:
        if self._on_event:
            try:
                self._on_event(event_type, path)
            except Exception:
                pass  # callbacks must never break the watcher loop

    def _block_until_signal(self) -> None:
        """Block the main thread; exit cleanly on SIGINT / SIGTERM."""
        stop_signals = (signal.SIGINT, signal.SIGTERM)

        def _handler(sig: int, frame: object) -> None:
            logger.info("Received signal %s, stopping watcher…", sig)
            self.stop()

        for sig in stop_signals:
            signal.signal(sig, _handler)

        logger.info("Watcher running. Press Ctrl+C to stop.")
        try:
            while self._running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            self.stop()
