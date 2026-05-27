"""Tests for embedding model."""

import pytest
from models.embeddings import EmbeddingModel


class TestEmbeddingModel:
    """Test suite for EmbeddingModel."""
    
    @pytest.fixture
    def model(self):
        """Load embedding model."""
        return EmbeddingModel(device="cpu")
    
    def test_encode_single_text(self, model):
        """Test encoding single text."""
        text = "This is a test movie"
        embedding = model.encode(text)
        
        assert embedding is not None
        assert len(embedding.shape) == 1
        assert embedding.shape[0] == model.embedding_dim
    
    def test_encode_multiple_texts(self, model):
        """Test encoding multiple texts."""
        texts = ["Movie 1", "Movie 2", "Movie 3"]
        embeddings = model.encode(texts)
        
        assert embeddings is not None
        assert len(embeddings) == 3
        assert embeddings[0].shape[0] == model.embedding_dim
    
    def test_similarity(self, model):
        """Test similarity computation."""
        text1 = "good movie"
        text2 = "good movie"
        
        emb1 = model.encode(text1)
        emb2 = model.encode(text2)
        
        similarity = model.similarity(emb1, emb2)
        assert 0.95 < similarity <= 1.0  # Should be very similar
