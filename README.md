# GenAI-Powered Movie Recommender System

A production-ready, AI-driven movie recommendation system leveraging PyTorch, transformers, and large language models for intelligent semantic understanding of user preferences.

##  Features

- **LLM-Based Embeddings**: Uses transformer models to understand semantic meaning of movie plots, reviews, and user preferences
- **Multi-Modal Learning**: Combines textual features with metadata (genres, ratings, cast)
- **Real-Time Inference**: Fast recommendation serving via FastAPI
- **Production-Ready**: Docker containerization, comprehensive logging, and monitoring
- **Scalable Architecture**: Designed for handling millions of movies and users
- **Evaluation Metrics**: NDCG, Recall@K, Precision@K, and diversity metrics
- **Fine-tuning Capabilities**: Custom model training on your domain data

##  Table of Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [API Documentation](#api-documentation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

##  Architecture

```
User Input (Preferences/History)
    ↓
Text Encoder (Sentence-Transformers/LLM)
    ↓
Embedding Generation
    ↓
Similarity Computation
    ↓
Ranking & Filtering
    ↓
Recommendation Output
```

### Components

1. **Embedding Layer**: Converts text to dense vectors using pre-trained transformers
2. **Indexing**: FAISS-based vector indexing for fast similarity search
3. **Ranking Module**: Re-ranks candidates using learned features
4. **Filtering**: Applies business logic, diversity constraints, and personalization
5. **API Server**: FastAPI-based REST interface for serving recommendations

##  Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (optional, for GPU acceleration)
- PostgreSQL 14+ (optional, for production data)

### Setup

```bash
# Clone repository
git clone https://github.com/rkratik/recommender-system.git
cd recommender-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your settings
```

##  Quick Start

### 1. Download Sample Data

```bash
python scripts/download_data.py --dataset moviedb
```

### 2. Train Embeddings

```bash
python train/train_embeddings.py --config configs/embeddings.yaml
```

### 3. Build Index

```bash
python scripts/build_index.py --model-path models/embeddings --output-path data/indices
```

### 4. Start API Server

```bash
uvicorn api.server:app --reload --port 8000
```

### 5. Get Recommendations

```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "num_recommendations": 10,
    "filters": {"min_rating": 6.5}
  }'
```

##  Project Structure

```
recommender-system/
├── api/
│   ├── __init__.py
│   ├── server.py           # FastAPI application
│   ├── schemas.py          # Request/response models
│   └── routes/
│       ├── recommendations.py
│       └── health.py
├── models/
│   ├── embeddings.py       # Embedding models
│   ├── ranker.py          # Ranking module
│   └── __init__.py
├── data/
│   ├── processors.py      # Data preprocessing
│   ├── loaders.py         # Data loading utilities
│   └── datasets/
├── train/
│   ├── train_embeddings.py
│   ├── train_ranker.py
│   └── callbacks.py
├── evaluation/
│   ├── metrics.py         # Evaluation metrics
│   ├── evaluator.py       # Evaluation runner
│   └── benchmark.py
├── scripts/
│   ├── download_data.py
│   ├── build_index.py
│   └── preprocess.py
├── configs/
│   ├── embeddings.yaml
│   ├── ranker.yaml
│   └── api.yaml
├── tests/
│   ├── test_embeddings.py
│   ├── test_api.py
│   └── test_metrics.py
├── logs/                   # Application logs
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── setup.py
├── .env.example
└── README.md
```

##  Model Details

### Embedding Model

- **Base**: `sentence-transformers/all-MiniLM-L6-v2` (production) or `all-mpnet-base-v2` (high-performance)
- **Dimension**: 384 or 768
- **Fine-tuning**: Optional domain-specific fine-tuning on movie metadata

### Ranking Model

- **Architecture**: 2-layer neural network on top of embeddings
- **Loss**: Contrastive loss with in-batch negatives
- **Optimization**: AdamW with warmup

##  API Documentation

### POST `/recommend`

Get personalized movie recommendations.

**Request:**
```json
{
  "user_id": "string",
  "user_history": ["movie_id_1", "movie_id_2"],
  "num_recommendations": 10,
  "filters": {
    "genres": ["Action", "Sci-Fi"],
    "min_rating": 6.5,
    "release_year_range": [2010, 2024]
  },
  "diversity": 0.5
}
```

**Response:**
```json
{
  "user_id": "string",
  "recommendations": [
    {
      "movie_id": "string",
      "title": "string",
      "score": 0.95,
      "reason": "Similar to movies you watched"
    }
  ],
  "timestamp": "2024-05-27T12:00:00Z"
}
```

### GET `/health`

Health check endpoint.

##  Training

### Train from Scratch

```bash
python train/train_embeddings.py \
  --config configs/embeddings.yaml \
  --data-path data/movies.csv \
  --output-dir models/custom
```

### Fine-tune Pre-trained

```bash
python train/train_embeddings.py \
  --config configs/embeddings.yaml \
  --pretrained-model sentence-transformers/all-mpnet-base-v2 \
  --fine-tune \
  --data-path data/movies.csv
```

##  Evaluation

### Run Benchmark

```bash
python evaluation/benchmark.py \
  --model-path models/embeddings \
  --test-data data/test.csv \
  --metrics "ndcg,recall,precision"
```

### Output

```
NDCG@10: 0.487
Recall@10: 0.523
Precision@10: 0.458
Diversity: 0.672
```

##  Docker Deployment

### Build Image

```bash
docker build -t genai-recommender:latest .
```

### Run Container

```bash
docker run -p 8000:8000 \
  -v /path/to/models:/app/models \
  -e OPENAI_API_KEY=your_key \
  genai-recommender:latest
```

### Docker Compose

```bash
docker-compose up -d
```

##  Configuration

Edit `configs/api.yaml` for API settings:

```yaml
server:
  host: 0.0.0.0
  port: 8000
  workers: 4

model:
  embedding_model: sentence-transformers/all-MiniLM-L6-v2
  index_type: faiss
  device: cuda

inference:
  batch_size: 32
  top_k: 100
  diversity_factor: 0.5

logging:
  level: INFO
  file: logs/api.log
```

##  Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_api.py::test_recommend -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

##  Performance

- **Embedding Generation**: ~1ms per sample
- **Vector Search**: ~10ms for 1M movies (FAISS)
- **Full Pipeline**: ~50ms per recommendation request
- **Throughput**: 100+ req/s per instance

## 📈 Future Enhancements

- [ ] Multi-modal embeddings (text + images)
- [ ] Real-time user preference updates
- [ ] A/B testing framework
- [ ] Explainability module (SHAP values)
- [ ] Cross-domain recommendations
- [ ] Graph neural networks for collaborative filtering
- [ ] Online learning with user feedback

##  Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see LICENSE file for details.

##  Support

For issues and questions:
- Open an issue on GitHub
- Check existing discussions
- Review documentation in `/docs`

