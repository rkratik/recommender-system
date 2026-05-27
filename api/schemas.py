"""Pydantic schemas for request/response validation."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class FilterParams(BaseModel):
    """Movie filter parameters."""
    genres: Optional[List[str]] = None
    min_rating: Optional[float] = Field(default=0.0, ge=0.0, le=10.0)
    max_rating: Optional[float] = Field(default=10.0, ge=0.0, le=10.0)
    release_year_range: Optional[tuple[int, int]] = None
    exclude_ids: Optional[List[str]] = None


class RecommendationRequest(BaseModel):
    """Request schema for recommendations."""
    user_id: str
    user_history: Optional[List[str]] = None
    user_preferences: Optional[str] = None
    num_recommendations: int = Field(default=10, ge=1, le=100)
    filters: Optional[FilterParams] = None
    diversity: float = Field(default=0.5, ge=0.0, le=1.0)
    diversity_weight: float = Field(default=0.3, ge=0.0, le=1.0)


class MovieScore(BaseModel):
    """Individual movie recommendation."""
    movie_id: str
    title: str
    score: float = Field(ge=0.0, le=1.0)
    reason: str
    genres: Optional[List[str]] = None
    rating: Optional[float] = None
    year: Optional[int] = None


class RecommendationResponse(BaseModel):
    """Response schema for recommendations."""
    user_id: str
    recommendations: List[MovieScore]
    count: int
    timestamp: datetime
    model_version: str = "0.1.0"


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    model_loaded: bool
    models: Dict[str, Any]


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: str
    timestamp: datetime
