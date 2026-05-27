"""Health check endpoints."""

from datetime import datetime
from fastapi import APIRouter, Depends
from api.schemas import HealthResponse
from api import server

router = APIRouter()


@router.get("/", response_model=HealthResponse)
async def health_check():
    """Check API health status."""
    embedding_loaded = server.embedding_model is not None
    ranker_loaded = server.ranker is not None
    
    return HealthResponse(
        status="healthy" if embedding_loaded and ranker_loaded else "degraded",
        timestamp=datetime.utcnow(),
        model_loaded=embedding_loaded and ranker_loaded,
        models={
            "embedding": "loaded" if embedding_loaded else "not loaded",
            "ranker": "loaded" if ranker_loaded else "not loaded"
        }
    )
