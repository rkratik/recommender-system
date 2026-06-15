"""Embedding model implementation using sentence-transformers with numpy fallback.

If torch / sentence-transformers are not installed (e.g. Python 3.14 where
PyTorch wheels are not yet published), the module falls back to a lightweight
numpy-based mock that still lets the API serve requests. Replace the mock with
the real SentenceTransformer once a compatible PyTorch wheel is available.
"""

import logging
import numpy as np
from typing import List, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy dependencies
# ---------------------------------------------------------------------------
try:
    import torch
    from sentence_transformers import SentenceTransformer
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    logger.warning(
        "torch / sentence_transformers not available. "
        "Using numpy mock embeddings. Install a compatible PyTorch wheel to "
        "enable real LLM-based embeddings."
    )


# ---------------------------------------------------------------------------
# Mock embedding model (numpy only)
# ---------------------------------------------------------------------------

class _MockEmbeddingModel:
    """Pure-numpy stand-in for SentenceTransformer.

    Produces deterministic, text-derived pseudo-embeddings using a simple
    hashing trick so that similar strings produce somewhat similar vectors.
    This is ONLY for development / deployment without a GPU-capable Python
    environment – it is NOT a real semantic model.
    """

    EMBEDDING_DIM = 384

    def get_sentence_embedding_dimension(self) -> int:
        return self.EMBEDDING_DIM

    def encode(
        self,
        sentences: List[str],
        batch_size: int = 32,
        convert_to_tensor: bool = False,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        embeddings = []
        for text in sentences:
            rng = np.random.default_rng(abs(hash(text)) % (2 ** 32))
            vec = rng.standard_normal(self.EMBEDDING_DIM).astype(np.float32)
            if normalize_embeddings:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
            embeddings.append(vec)
        return np.array(embeddings)


# ---------------------------------------------------------------------------
# Public EmbeddingModel class
# ---------------------------------------------------------------------------

class EmbeddingModel:
    """Embedding model for converting text to vectors.

    Uses sentence-transformers when available, otherwise falls back to a
    numpy-based mock so the service can start without ML dependencies.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ):
        """Initialise the embedding model.

        Args:
            model_name: HuggingFace model identifier (ignored in mock mode).
            device: Device to run the model on – ``'cuda'`` or ``'cpu'``.
        """
        self.model_name = model_name
        self.device = device

        if _HAS_TORCH:
            logger.info("Loading SentenceTransformer model '%s' on %s", model_name, device)
            try:
                self.model = SentenceTransformer(model_name, device=device)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                logger.info("Model loaded. Embedding dimension: %d", self.embedding_dim)
            except Exception as exc:
                logger.warning("Failed to load SentenceTransformer (%s). Using mock.", exc)
                self.model = _MockEmbeddingModel()
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
        else:
            logger.info("Running with mock numpy embeddings (dimension=%d)", _MockEmbeddingModel.EMBEDDING_DIM)
            self.model = _MockEmbeddingModel()
            self.embedding_dim = self.model.get_sentence_embedding_dimension()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        convert_to_tensor: bool = False,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        """Encode text(s) to embedding vectors.

        Args:
            texts: A single string or list of strings.
            batch_size: Batch size for encoding (used by real model).
            convert_to_tensor: If True and torch is available, return a Tensor.
            normalize_embeddings: Normalize output vectors to unit length.

        Returns:
            numpy array of shape ``(n_texts, embedding_dim)``, or a single
            vector of shape ``(embedding_dim,)`` when a single string is given.
        """
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        embeddings: np.ndarray = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=False,          # always get numpy first
            normalize_embeddings=normalize_embeddings,
        )

        # Optionally convert to torch Tensor if the real model is loaded
        if convert_to_tensor and _HAS_TORCH:
            import torch as _torch
            embeddings = _torch.tensor(embeddings)

        return embeddings[0] if single else embeddings

    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embedding vectors.

        Args:
            embedding1: First embedding vector.
            embedding2: Second embedding vector.

        Returns:
            Similarity score in ``[0, 1]``.
        """
        e1 = np.array(embedding1, dtype=np.float32)
        e2 = np.array(embedding2, dtype=np.float32)

        n1 = np.linalg.norm(e1)
        n2 = np.linalg.norm(e2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(e1, e2) / (n1 * n2))
