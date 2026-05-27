"""Evaluation metrics for recommendation systems."""

import logging
import numpy as np
from typing import List, Set

logger = logging.getLogger(__name__)


class RecommendationMetrics:
    """Metrics for evaluating recommendations."""
    
    @staticmethod
    def recall_at_k(relevant_items: Set, recommended_items: List, k: int = 10) -> float:
        """Calculate Recall@K.
        
        Args:
            relevant_items: Set of relevant items
            recommended_items: List of recommended items
            k: Number of recommendations
            
        Returns:
            Recall@K score
        """
        if len(relevant_items) == 0:
            return 0.0
        
        recommended_k = set(recommended_items[:k])
        intersection = len(relevant_items & recommended_k)
        return intersection / len(relevant_items)
    
    @staticmethod
    def precision_at_k(relevant_items: Set, recommended_items: List, k: int = 10) -> float:
        """Calculate Precision@K.
        
        Args:
            relevant_items: Set of relevant items
            recommended_items: List of recommended items
            k: Number of recommendations
            
        Returns:
            Precision@K score
        """
        recommended_k = set(recommended_items[:k])
        intersection = len(relevant_items & recommended_k)
        return intersection / k if k > 0 else 0.0
    
    @staticmethod
    def ndcg_at_k(relevant_items: Set, recommended_items: List, k: int = 10) -> float:
        """Calculate NDCG@K.
        
        Args:
            relevant_items: Set of relevant items
            recommended_items: List of recommended items (ordered)
            k: Number of recommendations
            
        Returns:
            NDCG@K score
        """
        dcg = 0.0
        for i, item in enumerate(recommended_items[:k]):
            if item in relevant_items:
                dcg += 1.0 / np.log2(i + 2)
        
        idcg = 0.0
        for i in range(min(k, len(relevant_items))):
            idcg += 1.0 / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def coverage(all_recommendations: List[List], total_items: int) -> float:
        """Calculate catalog coverage.
        
        Args:
            all_recommendations: All recommendations per user
            total_items: Total number of items in catalog
            
        Returns:
            Coverage score (0-1)
        """
        all_items = set()
        for recommendations in all_recommendations:
            all_items.update(recommendations)
        
        return len(all_items) / total_items if total_items > 0 else 0.0
