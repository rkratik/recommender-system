# Collaborative filtering recommendation using Surprise
# Assumes data/ratings.csv with columns: userId, movieId, rating

import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate

# Load data
data_path = 'data/ratings.csv'
df = pd.read_csv(data_path)

# Setup Surprise data format
reader = Reader(rating_scale=(df['rating'].min(), df['rating'].max()))
data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)

# Train SVD (matrix factorization) model
algo = SVD()
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)

# Fit model on full data
trainset = data.build_full_trainset()
algo.fit(trainset)

# Function to recommend top N movies for a user
def recommend_for_user(user_id, n=10):
    # Get list of all movieIds
    all_movies = df['movieId'].unique()
    # Movies seen by user
    seen = df[df['userId'] == user_id]['movieId'].unique()
    # Unseen movies
    unseen = [m for m in all_movies if m not in seen]
    predictions = [(mid, algo.predict(user_id, mid).est) for mid in unseen]
    top_n = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]
    return top_n

# Example usage
if __name__ == '__main__':
    user = 1  # Change as needed
    recs = recommend_for_user(user, n=5)
    print(f"Top recommendations for user {user}: {recs}")
