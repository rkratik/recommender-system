"""FastAPI server for movie recommendations."""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.routes import health, recommendations
from models.embeddings import EmbeddingModel
from models.ranker import Ranker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global state
embedding_model: Optional[EmbeddingModel] = None
ranker: Optional[Ranker] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    logger.info("Loading models...")
    global embedding_model, ranker
    
    try:
        embedding_model = EmbeddingModel()
        ranker = Ranker()
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="GenAI Movie Recommender API",
    description="Production-ready API for movie recommendations using LLM embeddings",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(recommendations.router, prefix="/recommend", tags=["Recommendations"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "GenAI Movie Recommender API",
        "version": "0.1.0",
        "status": "running"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
