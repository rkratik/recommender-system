"""Embedding model implementation using sentence-transformers."""

import logging
from typing import List, Union
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Embedding model for converting text to vectors."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize embedding model.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on (cuda/cpu)
        """
        self.model_name = model_name
        self.device = device
        logger.info(f"Loading embedding model {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        convert_to_tensor: bool = False,
        normalize_embeddings: bool = True
    ) -> Union[torch.Tensor, List[List[float]]]:
        """Encode text to embeddings.
        
        Args:
            texts: Text or list of texts to encode
            batch_size: Batch size for encoding
            convert_to_tensor: Return as torch tensor
            normalize_embeddings: Normalize embeddings to unit length
            
        Returns:
            Embeddings as tensor or list of lists
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=convert_to_tensor,
            normalize_embeddings=normalize_embeddings
        )
        
        return embeddings
    
    def similarity(self, embedding1, embedding2) -> float:
        """Compute cosine similarity between embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score (0-1)
        """
        return float(torch.nn.functional.cosine_similarity(
            torch.tensor(embedding1).unsqueeze(0),
            torch.tensor(embedding2).unsqueeze(0)
        ).item())
