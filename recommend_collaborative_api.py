from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
import pandas as pd
from recommend_collaborative import recommend_for_user, retrain_collaborative
import os

app = FastAPI(title="Collaborative Filtering Recommender API")

data_path = os.getenv('RATINGS_PATH', 'data/ratings.csv')

@app.get("/recommend_collaborative/")
def get_recommendations(user_id: int = Query(...), n: int = Query(10)):
    """Recommend top N movies for a given user using collaborative filtering."""
    try:
        recommendations = recommend_for_user(user_id, n)
        result = [
            {"movieId": int(mid), "predicted_rating": float(rating)}
            for mid, rating in recommendations
        ]
        return {"user_id": user_id, "recommendations": result}
    except Exception as e:
        return {"error": str(e)}

@app.post("/retrain_collaborative/")
def retrain():
    """Trigger retraining of the collaborative filtering model."""
    try:
        retrain_collaborative()
        return {"status": "success", "message": "Model retrained with latest ratings data."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# To run:
# uvicorn recommend_collaborative_api:app --reload
