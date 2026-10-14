"""
test_embedder.py – Tests for src/embedder.py.

Uses the real model (all-MiniLM-L6-v2) so we can verify actual vector shapes
and normalisation.  The model is cached after the first download, so CI runs
are fast after the first time.
"""

import numpy as np
import pytest

from src.embedder import Embedder


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    """Load the model once for the whole module (slow to load, fast to reuse)."""
    return Embedder()


class TestEmbedder:
    def test_output_shape(self, embedder):
        """embed() must return shape (N, dimension)."""
        texts = ["Hello world", "This is a test"]
        vecs = embedder.embed(texts)
        assert vecs.shape == (2, embedder.dimension)

    def test_dtype_is_float32(self, embedder):
        """FAISS requires float32 — ensure that's what we produce."""
        vecs = embedder.embed(["test"])
        assert vecs.dtype == np.float32

    def test_vectors_are_normalised(self, embedder):
        """L2 norm of every embedding should be 1.0 (±floating-point epsilon)."""
        vecs = embedder.embed(["Hello", "World", "Test sentence"])
        norms = np.linalg.norm(vecs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_embed_one_is_1d(self, embedder):
        """embed_one() should return a 1-D array, not (1, dim)."""
        vec = embedder.embed_one("single sentence")
        assert vec.ndim == 1
        assert vec.shape == (embedder.dimension,)

    def test_similar_texts_have_higher_score(self, embedder):
        """
        Cosine similarity of semantically similar texts should be higher
        than that of unrelated texts.
        """
        vec_a = embedder.embed_one("The cat sat on the mat")
        vec_b = embedder.embed_one("A cat is sitting on a rug")
        vec_c = embedder.embed_one("Quantum mechanics and wave functions")

        score_similar = float(np.dot(vec_a, vec_b))
        score_unrelated = float(np.dot(vec_a, vec_c))
        assert score_similar > score_unrelated

    def test_empty_list_raises(self, embedder):
        """embed([]) must raise ValueError, not silently return empty."""
        with pytest.raises(ValueError):
            embedder.embed([])

    def test_dimension_attribute(self, embedder):
        """The dimension attribute must match the actual vector size."""
        vecs = embedder.embed(["test"])
        assert vecs.shape[1] == embedder.dimension
