"""
test_vector_store.py – Tests for src/vector_store.py.

Uses random normalised vectors so we don't need the real embedding model.
Disk I/O tests use pytest's tmp_path fixture.
"""

import numpy as np
import pytest

from src.vector_store import VectorStore


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_vectors(n: int, dim: int = 384) -> np.ndarray:
    """Create n random L2-normalised float32 vectors of dimension dim."""
    rng = np.random.default_rng(seed=42)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def make_chunks(n: int) -> list[dict]:
    """Create n minimal chunk dicts."""
    return [
        {
            "chunk_text": f"This is chunk number {i}.",
            "file_name": "test.pdf",
            "page_num": i + 1,
            "chunk_index": i,
        }
        for i in range(n)
    ]


# ── Build ──────────────────────────────────────────────────────────────────────

class TestBuild:
    def test_is_ready_after_build(self):
        store = VectorStore()
        store.build(make_vectors(5), make_chunks(5))
        assert store.is_ready is True

    def test_total_chunks_correct(self):
        store = VectorStore()
        store.build(make_vectors(10), make_chunks(10))
        assert store.total_chunks == 10

    def test_mismatched_lengths_raise(self):
        store = VectorStore()
        with pytest.raises(ValueError):
            store.build(make_vectors(5), make_chunks(3))

    def test_empty_vectors_raise(self):
        store = VectorStore()
        with pytest.raises(ValueError):
            store.build(make_vectors(0), [])

    def test_not_ready_before_build(self):
        store = VectorStore()
        assert store.is_ready is False


# ── Add ────────────────────────────────────────────────────────────────────────

class TestAdd:
    def test_add_increases_total(self):
        store = VectorStore()
        store.build(make_vectors(5), make_chunks(5))
        store.add(make_vectors(3), make_chunks(3))
        assert store.total_chunks == 8

    def test_add_to_empty_store_builds(self):
        """add() on a fresh store should behave like build()."""
        store = VectorStore()
        store.add(make_vectors(4), make_chunks(4))
        assert store.is_ready is True
        assert store.total_chunks == 4


# ── Search ─────────────────────────────────────────────────────────────────────

class TestSearch:
    def test_returns_top_k_results(self):
        store = VectorStore()
        store.build(make_vectors(20), make_chunks(20))
        results = store.search(make_vectors(1)[0], top_k=4)
        assert len(results) == 4

    def test_results_contain_score(self):
        store = VectorStore()
        store.build(make_vectors(5), make_chunks(5))
        results = store.search(make_vectors(1)[0], top_k=2)
        for r in results:
            assert "score" in r
            assert isinstance(r["score"], float)

    def test_scores_are_in_valid_range(self):
        """Cosine similarity (inner product of unit vectors) ∈ [-1, 1]."""
        store = VectorStore()
        store.build(make_vectors(10), make_chunks(10))
        results = store.search(make_vectors(1)[0], top_k=5)
        for r in results:
            assert -1.0 <= r["score"] <= 1.0 + 1e-5  # small epsilon for float32

    def test_exact_match_scores_near_one(self):
        """
        Searching with a vector that is already in the index should return
        a score very close to 1.0 (perfect cosine similarity).
        """
        vecs = make_vectors(5)
        store = VectorStore()
        store.build(vecs, make_chunks(5))
        results = store.search(vecs[0], top_k=1)
        assert results[0]["score"] > 0.99

    def test_top_k_clamped_to_index_size(self):
        """Asking for more results than vectors should not error."""
        store = VectorStore()
        store.build(make_vectors(3), make_chunks(3))
        results = store.search(make_vectors(1)[0], top_k=10)
        assert len(results) == 3  # clamped to 3

    def test_empty_index_raises(self):
        store = VectorStore()
        with pytest.raises(RuntimeError):
            store.search(make_vectors(1)[0], top_k=4)

    def test_chunk_metadata_preserved_in_results(self):
        chunks = make_chunks(5)
        store = VectorStore()
        store.build(make_vectors(5), chunks)
        results = store.search(make_vectors(1)[0], top_k=1)
        assert "file_name" in results[0]
        assert "page_num" in results[0]
        assert "chunk_text" in results[0]


# ── Save / Load ────────────────────────────────────────────────────────────────

class TestSaveLoad:
    def test_save_and_load_roundtrip(self, tmp_path):
        """Saved index should load back with the same vectors and metadata."""
        idx_path = tmp_path / "index.faiss"
        meta_path = tmp_path / "chunks.pkl"

        vecs = make_vectors(8)
        chunks = make_chunks(8)

        store = VectorStore()
        store.build(vecs, chunks)
        store.save(idx_path, meta_path)

        store2 = VectorStore()
        loaded = store2.load(idx_path, meta_path)

        assert loaded is True
        assert store2.total_chunks == 8
        assert store2.is_ready is True

    def test_load_returns_false_if_missing(self, tmp_path):
        store = VectorStore()
        result = store.load(tmp_path / "no.faiss", tmp_path / "no.pkl")
        assert result is False

    def test_search_after_load_works(self, tmp_path):
        idx_path = tmp_path / "index.faiss"
        meta_path = tmp_path / "chunks.pkl"

        vecs = make_vectors(6)
        store = VectorStore()
        store.build(vecs, make_chunks(6))
        store.save(idx_path, meta_path)

        store2 = VectorStore()
        store2.load(idx_path, meta_path)
        results = store2.search(vecs[0], top_k=2)
        assert len(results) == 2


# ── Reset ──────────────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_clears_index(self, tmp_path):
        idx_path = tmp_path / "index.faiss"
        meta_path = tmp_path / "chunks.pkl"

        store = VectorStore()
        store.build(make_vectors(5), make_chunks(5))
        store.save(idx_path, meta_path)
        store.reset(idx_path, meta_path)

        assert store.is_ready is False
        assert not idx_path.exists()
        assert not meta_path.exists()


# ── get_indexed_files ──────────────────────────────────────────────────────────

class TestGetIndexedFiles:
    def test_returns_unique_file_names(self):
        chunks = [
            {"chunk_text": "a", "file_name": "a.pdf", "page_num": 1, "chunk_index": 0},
            {"chunk_text": "b", "file_name": "a.pdf", "page_num": 2, "chunk_index": 0},
            {"chunk_text": "c", "file_name": "b.pdf", "page_num": 1, "chunk_index": 0},
        ]
        store = VectorStore()
        store.build(make_vectors(3), chunks)
        files = store.get_indexed_files()
        assert sorted(files) == ["a.pdf", "b.pdf"]
