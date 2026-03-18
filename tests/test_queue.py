"""
Tests for the SQLiteIndexQueue in seekr.infrastructure.queue.index_queue.

The queue drives background indexing. These tests verify that:
  - files can be enqueued and retrieved
  - status transitions (pending → processing → done / failed) work correctly
  - statistics are accurate
  - the queue survives being re-opened from the same DB file
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from seekr.infrastructure.queue.index_queue import SQLiteIndexQueue
from seekr.domain.entities import QueueStats, TaskStatus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


@pytest.fixture()
def queue(tmp_db: Path) -> SQLiteIndexQueue:
    return SQLiteIndexQueue(db_path=tmp_db)


# ---------------------------------------------------------------------------
# Enqueue
# ---------------------------------------------------------------------------

class TestEnqueue:
    def test_enqueue_returns_positive_task_id(self, queue: SQLiteIndexQueue) -> None:
        task_id = queue.enqueue_file("/some/file.py", "abc123")
        assert task_id > 0

    def test_enqueued_task_is_pending(self, queue: SQLiteIndexQueue) -> None:
        queue.enqueue_file("/some/file.py", "abc123")
        tasks = queue.get_pending_tasks(limit=10)
        assert len(tasks) == 1
        assert tasks[0].status == TaskStatus.PENDING
        assert tasks[0].file_path == "/some/file.py"
        assert tasks[0].file_hash == "abc123"

    def test_multiple_files_enqueued(self, queue: SQLiteIndexQueue) -> None:
        for i in range(5):
            queue.enqueue_file(f"/file_{i}.txt", f"hash_{i}")
        tasks = queue.get_pending_tasks(limit=10)
        assert len(tasks) == 5

    def test_limit_parameter_is_respected(self, queue: SQLiteIndexQueue) -> None:
        for i in range(10):
            queue.enqueue_file(f"/file_{i}.txt", f"hash_{i}")
        tasks = queue.get_pending_tasks(limit=3)
        assert len(tasks) == 3


# ---------------------------------------------------------------------------
# Status transitions
# ---------------------------------------------------------------------------

class TestStatusTransitions:
    def test_mark_processing(self, queue: SQLiteIndexQueue) -> None:
        task_id = queue.enqueue_file("/a.py", "h1")
        queue.mark_processing(task_id)
        # Processing tasks no longer appear as pending
        assert queue.get_pending_tasks(limit=10) == []
        stats = queue.get_stats()
        assert stats.processing == 1

    def test_mark_done(self, queue: SQLiteIndexQueue) -> None:
        task_id = queue.enqueue_file("/a.py", "h1")
        queue.mark_processing(task_id)
        queue.mark_done(task_id)
        stats = queue.get_stats()
        assert stats.completed == 1
        assert stats.processing == 0
        assert stats.pending == 0

    def test_mark_failed(self, queue: SQLiteIndexQueue) -> None:
        task_id = queue.enqueue_file("/a.py", "h1")
        queue.mark_processing(task_id)
        queue.mark_failed(task_id)
        stats = queue.get_stats()
        assert stats.failed == 1
        assert stats.processing == 0


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

class TestQueueStats:
    def test_empty_queue_stats(self, queue: SQLiteIndexQueue) -> None:
        stats = queue.get_stats()
        assert stats.total == 0
        assert stats.pending == 0
        assert stats.completed == 0
        assert stats.processing == 0
        assert stats.failed == 0

    def test_progress_pct_is_zero_for_empty_queue(self, queue: SQLiteIndexQueue) -> None:
        assert queue.get_stats().progress_pct == 0.0

    def test_progress_pct_after_completion(self, queue: SQLiteIndexQueue) -> None:
        for i in range(4):
            tid = queue.enqueue_file(f"/f{i}.txt", f"h{i}")
            queue.mark_processing(tid)
            queue.mark_done(tid)
        stats = queue.get_stats()
        assert stats.progress_pct == 100.0

    def test_total_is_sum_of_all_statuses(self, queue: SQLiteIndexQueue) -> None:
        t1 = queue.enqueue_file("/a.py", "h1")
        t2 = queue.enqueue_file("/b.py", "h2")
        t3 = queue.enqueue_file("/c.py", "h3")
        queue.mark_processing(t1)
        queue.mark_done(t1)
        queue.mark_processing(t2)
        queue.mark_failed(t2)
        stats = queue.get_stats()
        assert stats.total == 3
        assert stats.completed == 1
        assert stats.failed == 1
        assert stats.pending == 1


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestQueuePersistence:
    def test_tasks_survive_queue_reinstantiation(self, tmp_db: Path) -> None:
        q1 = SQLiteIndexQueue(db_path=tmp_db)
        q1.enqueue_file("/persistent.txt", "hash_persist")

        q2 = SQLiteIndexQueue(db_path=tmp_db)
        tasks = q2.get_pending_tasks(limit=10)
        assert len(tasks) == 1
        assert tasks[0].file_path == "/persistent.txt"
