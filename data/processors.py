"""Data preprocessing utilities."""

import logging
import pandas as pd
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class DataProcessor:
    """Processor for movie data."""
    
    @staticmethod
    def clean_movies(df: pd.DataFrame) -> pd.DataFrame:
        """Clean movie dataframe.
        
        Args:
            df: Raw movie dataframe
            
        Returns:
            Cleaned dataframe
        """
        # Remove duplicates
        df = df.drop_duplicates(subset=['id'])
        
        # Remove movies with missing required fields
        required_fields = ['id', 'title']
        df = df.dropna(subset=required_fields)
        
        # Fill optional fields
        df['overview'] = df.get('overview', '').fillna('')
        df['rating'] = df.get('rating', 0).fillna(0)
        
        logger.info(f"Cleaned to {len(df)} movies")
        return df
    
    @staticmethod
    def split_data(
        df: pd.DataFrame,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train/val/test.
        
        Args:
            df: Input dataframe
            train_ratio: Training ratio
            val_ratio: Validation ratio
            test_ratio: Test ratio
            
        Returns:
            Tuple of (train, val, test) dataframes
        """
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train = df[:train_end]
        val = df[train_end:val_end]
        test = df[val_end:]
        
        logger.info(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
        return train, val, test
