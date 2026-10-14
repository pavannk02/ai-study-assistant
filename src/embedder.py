"""
embedder.py – Thin wrapper around sentence-transformers.

Uses the all-MiniLM-L6-v2 model which:
  - Runs 100% locally (no API calls, no internet needed after first download)
  - Downloads once (~90 MB) and is cached in ~/.cache/huggingface/
  - Produces 384-dimensional vectors
  - Is fast enough to embed thousands of chunks in seconds on CPU

Vectors are L2-normalised so that dot-product == cosine similarity.
This lets us use FAISS's IndexFlatIP (inner product) as a cosine index.
"""

import logging
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class Embedder:
    """
    Wraps SentenceTransformer and provides a simple embed() interface.

    The model is loaded once at construction time and reused for all calls.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        """
        Load the embedding model from the local cache (or download it).

        Args:
            model_name: HuggingFace model identifier.
                        Defaults to all-MiniLM-L6-v2 from config.
        """
        logger.info("Loading embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name)
        # get_embedding_dimension() is the new name in sentence-transformers 6.x;
        # fall back to the old name for compatibility with older versions.
        _get_dim = getattr(
            self._model,
            "get_embedding_dimension",
            getattr(self._model, "get_sentence_embedding_dimension", None),
        )
        self.dimension: int = _get_dim()
        logger.info("Embedding model ready. Vector dimension: %d", self.dimension)

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of strings and return L2-normalised float32 vectors.

        Normalisation means: ||v|| = 1 for every row, so
        dot-product(v_i, v_j) == cosine_similarity(v_i, v_j).

        Args:
            texts: Non-empty list of strings to embed.

        Returns:
            numpy array of shape (len(texts), self.dimension), dtype float32.

        Raises:
            ValueError: If texts is empty.
        """
        if not texts:
            raise ValueError("Cannot embed an empty list of texts.")

        logger.debug("Embedding %d texts…", len(texts))
        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,   # <-- L2 normalise
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        """
        Embed a single string and return a 1-D float32 array of shape (dim,).

        Convenience wrapper around embed() for query-time use.
        """
        return self.embed([text])[0]
