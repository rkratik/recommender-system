"""Data loaders for movie datasets."""

import logging
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class MovieDataLoader:
    """Loader for movie data."""
    
    def __init__(self, data_path: str):
        """Initialize loader.
        
        Args:
            data_path: Path to movie data file or directory
        """
        self.data_path = Path(data_path)
        self.movies = None
    
    def load_csv(self, filepath: str) -> pd.DataFrame:
        """Load movies from CSV.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame with movie data
        """
        logger.info(f"Loading movies from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} movies")
        return df
    
    def load_json(self, filepath: str) -> List[Dict[str, Any]]:
        """Load movies from JSON.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            List of movie dictionaries
        """
        logger.info(f"Loading movies from {filepath}")
        df = pd.read_json(filepath)
        logger.info(f"Loaded {len(df)} movies")
        return df.to_dict('records')
    
    def prepare_text(self, movie: Dict[str, Any]) -> str:
        """Prepare text representation of a movie.
        
        Args:
            movie: Movie dictionary
            
        Returns:
            Combined text for embedding
        """
        parts = []
        
        if 'title' in movie:
            parts.append(movie['title'])
        if 'overview' in movie or 'description' in movie:
            parts.append(movie.get('overview') or movie.get('description', ''))
        if 'genres' in movie:
            genres = movie['genres']
            if isinstance(genres, list):
                parts.append(' '.join(genres))
            else:
                parts.append(str(genres))
        
        return ' '.join(parts)
