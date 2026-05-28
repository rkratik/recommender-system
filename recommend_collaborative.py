# Collaborative filtering recommendation using Surprise
# Assumes data/ratings.csv with columns: userId, movieId, rating

import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate

# Globals for model and data
data_path = 'data/ratings.csv'
df = None
algo = None
reader = None
data = None

def load_data():
    global df, reader, data
    df = pd.read_csv(data_path)
    reader = Reader(rating_scale=(df['rating'].min(), df['rating'].max()))
    data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)

# Train SVD (matrix factorization) model
def train_model():
    global algo
    if df is None or data is None:
        load_data()
    algo = SVD()
    cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)
    trainset = data.build_full_trainset()
    algo.fit(trainset)

# Call on import
load_data()
train_model()

def recommend_for_user(user_id, n=10):
    if algo is None or df is None:
        train_model()
    all_movies = df['movieId'].unique()
    seen = df[df['userId'] == user_id]['movieId'].unique()
    unseen = [m for m in all_movies if m not in seen]
    predictions = [(mid, algo.predict(user_id, mid).est) for mid in unseen]
    top_n = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]
    return top_n

# API-accessible retrain

def retrain_collaborative():
    load_data()
    train_model()
    return True

if __name__ == '__main__':
    user = 1  # Change as needed
    recs = recommend_for_user(user, n=5)
    print(f"Top recommendations for user {user}: {recs}")
    print('Retraining...')
    retrain_collaborative()
    print('Retrained!')
