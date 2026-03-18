"""
FAISSVectorStore — FAISS-backed implementation of the VectorStore port.

Design decisions:
  - Uses IndexFlatIP (exact inner product) because vectors are unit-normalised,
    making inner product equivalent to cosine similarity.
  - Wraps with IndexIDMap2 so we can associate integer IDs with chunk_id strings.
  - Chunk_id strings are mapped to int64 IDs via an in-memory dict; the mapping
    is persisted alongside the FAISS index in a companion JSON file.
  - For 32 GB RAM systems with up to ~500 k chunks the flat index fits comfortably.
    For 1 M+ chunks, swap to IndexHNSWFlat (see SCALABILITY note below).
  - Thread safety: FAISS is not thread-safe for writes. A threading.Lock guards
    all mutating operations.

SCALABILITY note:
  At 100 k+ files, replace IndexFlatIP with IndexHNSWFlat(dim, 32) for
  sub-linear search time.  The interface is identical — only this file changes.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np

from seekr.domain.exceptions import StoreError
from seekr.domain.interfaces import VectorStore

logger = logging.getLogger(__name__)


class FAISSVectorStore(VectorStore):
    """
    Concrete VectorStore implementation backed by FAISS.

    State is persisted to two files:
      - <store_dir>/index.faiss  — the FAISS binary index
      - <store_dir>/id_map.json  — chunk_id ↔ int64 ID mapping
    """

    def __init__(self, store_dir: Path, dimension: int) -> None:
        """
        Args:
            store_dir:  Directory where index files are saved.
            dimension:  Embedding dimension (must match the EmbeddingModel).
        """
        self._store_dir = store_dir
        self._dim = dimension
        self._lock = threading.Lock()

        # Maps chunk_id (str) → internal int64 ID
        self._id_map: dict[str, int] = {}
        # Reverse map: int64 ID → chunk_id
        self._rev_map: dict[int, str] = {}
        self._next_id: int = 0

        self._index: Optional[Any] = None  # lazy-initialised; faiss has no type stubs
        self._store_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    @property
    def _index_path(self) -> Path:
        return self._store_dir / "index.faiss"

    @property
    def _id_map_path(self) -> Path:
        return self._store_dir / "id_map.json"

    # ------------------------------------------------------------------
    # VectorStore interface
    # ------------------------------------------------------------------

    def add(self, chunk_ids: list[str], vectors: list[list[float]]) -> None:
        """Add or overwrite vectors."""
        if len(chunk_ids) != len(vectors):
            raise StoreError("chunk_ids and vectors must have equal length.")
        if not chunk_ids:
            return

        arr = np.array(vectors, dtype=np.float32)
        if arr.shape[1] != self._dim:
            raise StoreError(
                f"Vector dimension mismatch: expected {self._dim}, got {arr.shape[1]}"
            )

        with self._lock:
            index = self._get_index()

            int_ids: list[int] = []
            for cid in chunk_ids:
                if cid in self._id_map:
                    # Remove old vector before re-adding (IDMap2 supports this)
                    old_id = self._id_map[cid]
                    try:
                        index.remove_ids(np.array([old_id], dtype=np.int64))
                    except Exception:  # noqa: BLE001
                        pass  # may not exist if index was freshly rebuilt

                new_id = self._next_id
                self._next_id += 1
                self._id_map[cid] = new_id
                self._rev_map[new_id] = cid
                int_ids.append(new_id)

            index.add_with_ids(arr, np.array(int_ids, dtype=np.int64))
            logger.debug("Added %d vectors to FAISS index.", len(chunk_ids))

    def search(self, query_vector: list[float], top_k: int) -> list[tuple[str, float]]:
        """Return top_k nearest neighbours as (chunk_id, score) pairs."""
        with self._lock:
            index = self._get_index()
            if index.ntotal == 0:
                return []

            q = np.array([query_vector], dtype=np.float32)
            effective_k = min(top_k, index.ntotal)
            distances, ids = index.search(q, effective_k)

            results: list[tuple[str, float]] = []
            for dist, int_id in zip(distances[0], ids[0]):
                if int_id == -1:
                    continue
                chunk_id = self._rev_map.get(int(int_id))
                if chunk_id is not None:
                    results.append((chunk_id, float(dist)))
            return results

    def delete(self, chunk_ids: list[str]) -> None:
        """Remove vectors by chunk_id."""
        if not chunk_ids:
            return
        with self._lock:
            index = self._get_index()
            to_remove: list[int] = []
            for cid in chunk_ids:
                int_id = self._id_map.pop(cid, None)
                if int_id is not None:
                    self._rev_map.pop(int_id, None)
                    to_remove.append(int_id)
            if to_remove:
                index.remove_ids(np.array(to_remove, dtype=np.int64))
                logger.debug("Deleted %d vectors from FAISS index.", len(to_remove))

    def persist(self) -> None:
        """Flush index and ID mapping to disk."""
        with self._lock:
            import faiss  # noqa: PLC0415

            if self._index is not None:
                faiss.write_index(self._index, str(self._index_path))

            mapping = {
                "id_map": self._id_map,
                "rev_map": {str(k): v for k, v in self._rev_map.items()},
                "next_id": self._next_id,
            }
            self._id_map_path.write_text(json.dumps(mapping), encoding="utf-8")
            logger.debug("FAISS index persisted to %s", self._store_dir)

    def load(self) -> None:
        """Load persisted index and ID mapping from disk."""
        with self._lock:
            import faiss  # noqa: PLC0415

            if self._index_path.exists():
                self._index = faiss.read_index(str(self._index_path))
                logger.debug(
                    "Loaded FAISS index: %d vectors", self._index.ntotal
                )
            else:
                self._index = self._make_empty_index()
                logger.debug("No existing FAISS index found; starting fresh.")

            if self._id_map_path.exists():
                data = json.loads(self._id_map_path.read_text(encoding="utf-8"))
                self._id_map = data.get("id_map", {})
                self._rev_map = {int(k): v for k, v in data.get("rev_map", {}).items()}
                self._next_id = data.get("next_id", 0)

    @property
    def total_vectors(self) -> int:
        with self._lock:
            if self._index is None:
                return 0
            return int(self._index.ntotal)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_index(self) -> Any:
        """Return existing index or create a new one."""
        if self._index is None:
            self._index = self._make_empty_index()
        return self._index

    def _make_empty_index(self) -> Any:
        """Create an empty FAISS IDMap2 wrapping a flat inner-product index."""
        try:
            import faiss  # noqa: PLC0415

            flat = faiss.IndexFlatIP(self._dim)
            index = faiss.IndexIDMap2(flat)
            return index
        except ImportError as exc:
            raise StoreError(
                "faiss-cpu is not installed. Run: pip install faiss-cpu"
            ) from exc
