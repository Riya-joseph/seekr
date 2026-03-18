"""
Background index workers — process queued indexing tasks.

Two modes:
  - Drain mode (default): workers exit when the queue is empty.
    Used by `seekr index` to process a bounded set of tasks.
  - Watch mode (stop_event given): workers poll the queue until the
    stop_event is set. Used by `seekr watch` so workers stay alive
    between file events and process tasks as they arrive.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from seekr.config.settings import NUM_INDEX_WORKERS

if TYPE_CHECKING:
    from seekr.application.index_service import IndexService
    from seekr.domain.interfaces import IndexQueue

logger = logging.getLogger(__name__)

_POLL_INTERVAL: float = 0.5  # seconds between queue checks when idle


def run_worker(
    queue: IndexQueue,
    index_service: IndexService,
    stop_event: Optional[threading.Event] = None,
    ignore_patterns: Optional[set[str]] = None,
) -> None:
    """
    Run a single worker loop.

    - If stop_event is None (drain mode): exits when no pending tasks remain.
    - If stop_event is given (watch mode): polls every _POLL_INTERVAL seconds
      until stop_event is set, processing tasks as they arrive.

    Args:
        ignore_patterns: Path-component patterns to skip (e.g. from --exclude).
                         Passed to index_file so that files from excluded
                         directories are not indexed even when dequeued from
                         tasks that were queued in a previous run.
    """
    logger.debug("Worker started (mode=%s)", "watch" if stop_event else "drain")
    while True:
        # Respect stop signal first so shutdown is immediate even under load
        if stop_event is not None and stop_event.is_set():
            logger.debug("Worker received stop signal, exiting.")
            break

        tasks = queue.get_pending_tasks(limit=1)
        if not tasks:
            if stop_event is None:
                break  # Drain mode: nothing left to do
            time.sleep(_POLL_INTERVAL)
            continue

        task = tasks[0]
        path = Path(task.file_path)
        logger.info("Processing task id=%d file=%s", task.id, path)
        queue.mark_processing(task.id)
        try:
            result = index_service.index_file(path, ignore_patterns=ignore_patterns)
            if result in ("indexed", "skipped"):
                queue.mark_done(task.id)
                logger.debug("Task done id=%d result=%s", task.id, result)
            else:
                record = index_service.get_record(str(path))
                err = record.error_message if record and record.error_message else "unknown"
                logger.warning("Task failed id=%d file=%s error=%s", task.id, path, err)
                queue.mark_failed(task.id)
        except Exception as exc:
            logger.exception("Task failed id=%d file=%s error=%s", task.id, path, exc)
            queue.mark_failed(task.id)


def run_worker_pool(
    queue: IndexQueue,
    index_service: IndexService,
    num_workers: int = NUM_INDEX_WORKERS,
    daemon: bool = True,
    stop_event: Optional[threading.Event] = None,
    ignore_patterns: Optional[set[str]] = None,
) -> Optional[list[threading.Thread]]:
    """
    Start a pool of worker threads.

    Args:
        queue:           The indexing task queue.
        index_service:   Sync IndexService (no queue) used by workers.
        num_workers:     Number of parallel worker threads.
        daemon:          If True (default), threads do not block process exit.
                         If False, blocks until queue is drained (drain mode only).
        stop_event:      Pass a threading.Event to run in watch mode (polling).
                         Workers stay alive until stop_event.set() is called.
                         When None, workers exit when the queue is empty (drain mode).
        ignore_patterns: Path-component patterns forwarded to every worker so
                         excluded files are never indexed regardless of what is
                         already in the queue.

    Returns:
        List of started threads when in watch mode (stop_event given);
        None in drain mode (callers block inside this function until done).
    """
    logger.info(
        "Starting index worker pool workers=%d mode=%s",
        num_workers,
        "watch" if stop_event else "drain",
    )

    threads: list[threading.Thread] = []
    for _ in range(num_workers):
        t = threading.Thread(
            target=run_worker,
            args=(queue, index_service, stop_event, ignore_patterns),
            daemon=daemon,
        )
        t.start()
        threads.append(t)

    if not daemon and stop_event is None:
        # Drain mode: block until all workers finish
        for t in threads:
            t.join()
        logger.debug("Worker pool drained")
        return None

    return threads
