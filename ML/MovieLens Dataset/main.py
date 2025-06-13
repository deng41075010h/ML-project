import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

# --- 1. 讀取資料與 genre multi-hot 編碼 ---
def load_movie_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    movies["genres"] = movies["genres"].apply(lambda x: x.split("|"))

    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(movies["genres"])
    genre_df = pd.DataFrame(genre_matrix, columns=mlb.classes_, index=movies["movieId"])
    return movies, ratings, genre_df, mlb.classes_

# --- 2. 每個使用者分開做 train/test split ---
def train_test_split_per_user(ratings, test_ratio=0.2, seed=42):
    np.random.seed(seed)
    train_list, test_list = [], []

    for user_id, group in ratings.groupby("userId"):
        group = group.sample(frac=1, random_state=seed)
        test_size = int(len(group) * test_ratio)
        test = group.iloc[:test_size]
        train = group.iloc[test_size:]

        test_list.append(test)
        train_list.append(train)

    return pd.concat(train_list), pd.concat(test_list)

# --- 3. 加入時間衰退權重 (time decay) ---
def compute_time_weights(timestamps, base_time=None, alpha=0.001):
    if base_time is None:
        base_time = timestamps.max()
    time_diff = base_time - timestamps
    return np.exp(-alpha * time_diff)

# --- 4. 建立使用者偏好向量（支援時間偏差） ---
def build_user_profiles(train_ratings, genre_df, mlb_classes, use_time_bias=False, alpha=0.001):
    merged = train_ratings.merge(genre_df, left_on="movieId", right_index=True)

    if use_time_bias:
        base_time = merged["timestamp"].max()
        merged["time_weight"] = compute_time_weights(merged["timestamp"], base_time, alpha)
        merged["weighted_rating"] = merged["rating"] * merged["time_weight"]
        weight_col = "weighted_rating"
    else:
        merged["weighted_rating"] = merged["rating"]
        weight_col = "weighted_rating"

    return merged.groupby("userId").apply(
        lambda x: (x[mlb_classes].T @ x[weight_col]).div(x[weight_col].sum())
    )

# --- 5. 手刻餘弦相似度計算 ---
def cosine_similarity_manual(u, v):
    dot = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 0.0
    return dot / (norm_u * norm_v)

# --- 6. 計算 Hit Rate@K ---
def hit_rate_at_k(user_profiles, test_ratings, train_ratings, genre_df, k=10):
    hits = 0
    total_users = 0

    genre_matrix = genre_df.values
    movie_ids = genre_df.index.tolist()

    for user_id in test_ratings["userId"].unique():
        if user_id not in user_profiles.index:
            continue

        user_vector = user_profiles.loc[user_id].values
        seen = set(train_ratings[train_ratings["userId"] == user_id]["movieId"])

        similarities = []
        for idx, mid in enumerate(movie_ids):
            if mid in seen:
                continue
            movie_vector = genre_matrix[idx]
            sim = cosine_similarity_manual(user_vector, movie_vector)
            similarities.append((mid, sim))

        recommendations = sorted(similarities, key=lambda x: -x[1])[:k]
        recommended_ids = set(mid for mid, _ in recommendations)

        test_movies = set(test_ratings[test_ratings["userId"] == user_id]["movieId"])
        if recommended_ids & test_movies:
            hits += 1

        total_users += 1

    return hits / total_users if total_users else 0

# --- 7. 主程式入口 ---
if __name__ == "__main__":
    use_time_bias = False   # ← 若不想用時間權重，改為 False
    alpha = 0.001          # ← 時間衰退參數（越大越快衰退）
    top_k = 10             # ← 推薦前幾名用來算 HR@K

    print("Loading data...")
    movies, ratings, genre_df, mlb_classes = load_movie_data()

    print("Splitting train/test...")
    train_ratings, test_ratings = train_test_split_per_user(ratings, test_ratio=0.2)

    print("Building user profiles...")
    user_profiles = build_user_profiles(train_ratings, genre_df, mlb_classes, use_time_bias, alpha)

    print("\nCBF without time bias")
    print(f"Evaluating Hit Rate@{top_k} (use_time_bias={False})...")
    hr = hit_rate_at_k(user_profiles, test_ratings, train_ratings, genre_df, k=top_k)
    print(f"Hit Rate@{top_k} = {hr:.4f}\n")

    print("CBF with time bias")
    print(f"Evaluating Hit Rate@{top_k} (use_time_bias={True})...")
    hr = hit_rate_at_k(user_profiles, test_ratings, train_ratings, genre_df, k=top_k)
    print(f"Hit Rate@{top_k} = {hr:.4f}")
