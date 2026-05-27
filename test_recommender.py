"""Test script for recommender system."""

import pandas as pd
from models.embeddings import EmbeddingModel
from sklearn.metrics.pairwise import cosine_similarity

print("=" * 60)
print("🚀 Recommender System Test Script")
print("=" * 60)

# Step 1: Load embedding model
print("\n1️⃣ Loading embedding model...")
try:
    embedding_model = EmbeddingModel(device="cpu")
    print(f"   ✓ Model loaded successfully!")
    print(f"   Embedding dimension: {embedding_model.embedding_dim}")
except Exception as e:
    print(f"   ✗ Error loading model: {e}")
    exit(1)

# Step 2: Load movies
print("\n2️⃣ Loading movie dataset...")
try:
    movies = pd.read_csv('data/movies/ml-1m/movies.dat', 
                          sep='::', 
                          header=None, 
                          names=['id', 'title', 'genres'],
                          encoding='latin-1')
    print(f"   ✓ Loaded {len(movies)} movies!")
    print(f"\n   Sample movies:")
    for idx, row in movies.head(5).iterrows():
        print(f"      - {row['title']} ({row['genres']})")
except Exception as e:
    print(f"   ✗ Error loading movies: {e}")
    print(f"   Make sure you have downloaded the dataset:")
    print(f"   data/movies/ml-1m/movies.dat")
    exit(1)

# Step 3: Create embeddings for sample movies
print("\n3️⃣ Creating embeddings for movies...")
try:
    # Create text representation
    sample_movies = movies.head(100).copy()
    texts = [f"{row['title']} {row['genres']}" for _, row in sample_movies.iterrows()]
    
    print(f"   Processing {len(texts)} movies...")
    embeddings = embedding_model.encode(texts, batch_size=32, normalize_embeddings=True)
    print(f"   ✓ Embeddings created! Shape: {embeddings.shape}")
except Exception as e:
    print(f"   ✗ Error creating embeddings: {e}")
    exit(1)

# Step 4: Get a sample movie
print("\n4️⃣ Selecting a sample movie...")
sample_idx = 0
sample_movie = movies.iloc[sample_idx]
print(f"   📽️  {sample_movie['title']}")
print(f"   Genres: {sample_movie['genres']}")

# Step 5: Find similar movies
print("\n5️⃣ Finding similar movies...")
try:
    sample_text = f"{sample_movie['title']} {sample_movie['genres']}"
    sample_embedding = embedding_model.encode(sample_text, normalize_embeddings=True)
    
    # Compute similarities
    similarities = cosine_similarity([sample_embedding], embeddings)[0]
    top_indices = similarities.argsort()[-10:][::-1]
    
    print(f"\n   🎬 Top 10 Similar Movies:")
    for rank, idx in enumerate(top_indices, 1):
        movie = sample_movies.iloc[idx]
        similarity = similarities[idx]
        print(f"   {rank:2d}. {movie['title']:50s} (score: {similarity:.4f})")
except Exception as e:
    print(f"   ✗ Error finding similar movies: {e}")
    exit(1)

# Step 6: Test the API
print("\n6️⃣ Testing API connection...")
try:
    import requests
    response = requests.get('http://localhost:8000/health', timeout=5)
    if response.status_code == 200:
        print(f"   ✓ API is running! Status: {response.status_code}")
        health_data = response.json()
        print(f"   Health: {health_data.get('status')}")
    else:
        print(f"   ⚠️  API returned status: {response.status_code}")
except requests.exceptions.ConnectionError:
    print(f"   ⚠️  API is not running. Start it with:")
    print(f"      uvicorn api.server:app --reload --port 8000")
except Exception as e:
    print(f"   ⚠️  Error testing API: {e}")

print("\n" + "=" * 60)
print("✅ Test completed successfully!")
print("=" * 60)
print("\nNext steps:")
print("1. Download full dataset: python scripts/download_data.py")
print("2. Train embeddings: python train/train_embeddings.py")
print("3. Build FAISS index: python scripts/build_index.py")
print("4. Test API: curl http://localhost:8000/docs")
print("=" * 60)
