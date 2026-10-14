"""
vector_store.py – Build, save, load, and search a FAISS index.

Design decisions:
  - IndexFlatIP (inner product): exact cosine search when vectors are normalised.
    No approximation, no hyperparameters to tune — ideal for small-to-medium
    corpora (up to ~100 k chunks) and easy to explain in an interview.
  - The chunk metadata (list of ChunkData dicts) is stored alongside the index
    in a pickle file.  FAISS itself only stores raw vectors, not text.
  - Both files live in data/ (gitignored) so the index survives Streamlit restarts
    without re-embedding.

Cosine similarity score range: [-1, 1].  Higher is more similar.
"""

import logging
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from src.config import FAISS_INDEX_PATH, CHUNKS_METADATA_PATH

logger = logging.getLogger(__name__)

# Type alias (same as in chunker.py — repeated here to avoid a circular import)
ChunkData = dict[str, Any]


class VectorStore:
    """
    Wraps a FAISS IndexFlatIP and the parallel list of chunk metadata.

    Usage:
        store = VectorStore()
        store.build(vectors, chunks)
        store.save()

        # --- on next run ---
        store = VectorStore()
        store.load()
        results = store.search(query_vector, top_k=4)
    """

    def __init__(self) -> None:
        self._index: faiss.IndexFlatIP | None = None
        self._chunks: list[ChunkData] = []

    # ── Build ──────────────────────────────────────────────────────────────────

    def build(self, vectors: np.ndarray, chunks: list[ChunkData]) -> None:
        """
        Create a new FAISS index from pre-computed (normalised) vectors.

        Args:
            vectors: float32 array of shape (N, dim).  Must be L2-normalised.
            chunks:  List of N ChunkData dicts — one per row in vectors.

        Raises:
            ValueError: If vectors and chunks lengths don't match.
        """
        if len(vectors) != len(chunks):
            raise ValueError(
                f"vectors ({len(vectors)}) and chunks ({len(chunks)}) must have the same length."
            )
        if len(vectors) == 0:
            raise ValueError("Cannot build an index from zero vectors.")

        dimension = vectors.shape[1]
        logger.info(
            "Building FAISS IndexFlatIP: %d vectors, dim=%d.", len(vectors), dimension
        )

        self._index = faiss.IndexFlatIP(dimension)
        self._index.add(vectors)
        self._chunks = list(chunks)  # make a copy so callers can't mutate it

        logger.info("Index built. Total vectors: %d.", self._index.ntotal)

    def add(self, vectors: np.ndarray, chunks: list[ChunkData]) -> None:
        """
        Add more vectors to an existing index (used when new PDFs are uploaded).

        Args:
            vectors: float32 array of shape (M, dim).  Must match index dimension.
            chunks:  List of M ChunkData dicts.
        """
        if self._index is None:
            self.build(vectors, chunks)
            return

        if len(vectors) != len(chunks):
            raise ValueError("vectors and chunks must have the same length.")

        self._index.add(vectors)
        self._chunks.extend(chunks)
        logger.info(
            "Added %d vectors. Total in index: %d.", len(vectors), self._index.ntotal
        )

    # ── Persist ────────────────────────────────────────────────────────────────

    def save(
        self,
        index_path: Path = FAISS_INDEX_PATH,
        meta_path: Path = CHUNKS_METADATA_PATH,
    ) -> None:
        """
        Save the FAISS index and chunk metadata to disk.

        Args:
            index_path: Where to write the .faiss file.
            meta_path:  Where to write the pickled chunk list.
        """
        if self._index is None:
            raise RuntimeError("Nothing to save — build() or load() first.")

        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(index_path))

        with open(meta_path, "wb") as f:
            pickle.dump(self._chunks, f)

        logger.info("Index saved → %s (%d vectors).", index_path, self._index.ntotal)

    def load(
        self,
        index_path: Path = FAISS_INDEX_PATH,
        meta_path: Path = CHUNKS_METADATA_PATH,
    ) -> bool:
        """
        Load index and metadata from disk.

        Returns:
            True if loaded successfully, False if either file is missing.
        """
        if not index_path.exists() or not meta_path.exists():
            logger.info("No saved index found at %s.", index_path)
            return False

        self._index = faiss.read_index(str(index_path))

        with open(meta_path, "rb") as f:
            self._chunks = pickle.load(f)

        logger.info(
            "Index loaded from %s (%d vectors).", index_path, self._index.ntotal
        )
        return True

    # ── Search ─────────────────────────────────────────────────────────────────

    def search(
        self, query_vector: np.ndarray, top_k: int = 4
    ) -> list[dict[str, Any]]:
        """
        Find the top-k most similar chunks for a query vector.

        Args:
            query_vector: 1-D float32 array of shape (dim,). Must be L2-normalised.
            top_k:        Number of results to return.

        Returns:
            List of dicts, each containing all ChunkData fields plus:
              ``score`` (float): cosine similarity in [-1, 1], higher is better.

        Raises:
            RuntimeError: If the index is empty.
        """
        if self._index is None or self._index.ntotal == 0:
            raise RuntimeError("Index is empty. Upload and process some PDFs first.")

        # FAISS expects a 2-D array: (num_queries, dim)
        query_2d = query_vector.reshape(1, -1).astype(np.float32)

        # Clamp top_k to the number of stored vectors
        k = min(top_k, self._index.ntotal)

        scores, indices = self._index.search(query_2d, k)

        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue  # FAISS returns -1 for padding when k > ntotal
            result = dict(self._chunks[idx])
            result["score"] = float(score)
            results.append(result)

        return results

    # ── Introspection ──────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        """True if the index has been built or loaded and contains vectors."""
        return self._index is not None and self._index.ntotal > 0

    @property
    def total_chunks(self) -> int:
        """Number of chunks currently in the index."""
        return self._index.ntotal if self._index else 0

    def get_indexed_files(self) -> list[str]:
        """Return a deduplicated list of file names currently in the index."""
        return list(dict.fromkeys(c["file_name"] for c in self._chunks))

    def reset(
        self,
        index_path: Path = FAISS_INDEX_PATH,
        meta_path: Path = CHUNKS_METADATA_PATH,
    ) -> None:
        """
        Clear the in-memory index and delete saved files from disk.
        Called when the user clicks 'Reset index' in the UI.
        """
        self._index = None
        self._chunks = []
        if index_path.exists():
            index_path.unlink()
        if meta_path.exists():
            meta_path.unlink()
        logger.info("Index reset. Saved files deleted.")
