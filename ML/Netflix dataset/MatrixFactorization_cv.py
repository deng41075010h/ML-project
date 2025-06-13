import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from collections import defaultdict
from DataProcessing import load_netflix_data, filter_sparse, train_test_split_strict, verify_train_test_overlap

# === Matrix Factorization with Bias ===
def matrix_factorization(train, n_factors=20, n_iters=10, lr=0.01, reg=0.01):
    user_ids = train['user_id'].unique()
    movie_ids = train['movie_id'].unique()
    user_map = {u: i for i, u in enumerate(user_ids)}
    movie_map = {m: i for i, m in enumerate(movie_ids)}

    n_users, n_movies = len(user_ids), len(movie_ids)
    P = np.random.normal(0, 0.1, (n_users, n_factors))
    Q = np.random.normal(0, 0.1, (n_movies, n_factors))
    user_bias = np.zeros(n_users)
    movie_bias = np.zeros(n_movies)
    global_mean = train['rating'].mean()

    for iter_num in range(1, n_iters + 1):
        for _, row in train.iterrows():
            u = user_map[row['user_id']]
            m = movie_map[row['movie_id']]
            r = row['rating']

            pred = global_mean + user_bias[u] + movie_bias[m] + np.dot(P[u], Q[m])
            err = r - pred

            user_bias[u] += lr * (err - reg * user_bias[u])
            movie_bias[m] += lr * (err - reg * movie_bias[m])
            P[u] += lr * (err * Q[m] - reg * P[u])
            Q[m] += lr * (err * P[u] - reg * Q[m])

        preds = []
        truths = []
        for _, row in train.iterrows():
            u = user_map[row['user_id']]
            m = movie_map[row['movie_id']]
            pred = global_mean + user_bias[u] + movie_bias[m] + np.dot(P[u], Q[m])
            preds.append(pred)
            truths.append(row['rating'])

        rmse = np.sqrt(mean_squared_error(truths, preds))
        print(f"Iteration {iter_num}/{n_iters}, RMSE = {rmse:.4f}")

    return P, Q, user_bias, movie_bias, global_mean, user_map, movie_map


# === Evaluation ===
def evaluate_rmse_mf(test_df, P, Q, user_bias, movie_bias, global_mean, user_map, movie_map):
    preds = []
    actuals = []
    for _, row in test_df.iterrows():
        uid, mid = row['user_id'], row['movie_id']
        if uid in user_map and mid in movie_map:
            u, m = user_map[uid], movie_map[mid]
            pred = global_mean + user_bias[u] + movie_bias[m] + np.dot(P[u], Q[m])
            preds.append(pred)
            actuals.append(row['rating'])
    return np.sqrt(mean_squared_error(actuals, preds))

# === Hit Rate Calculation ===
def build_user_item_dict(df):
    user_item = {}
    for _, row in df.iterrows():
        user_item.setdefault(row['user_id'], set()).add(row['movie_id'])
    return user_item

def recommend_mf(P, Q, user_bias, movie_bias, global_mean, user_map, train_df, movie_map, user_id, N=10):
    if user_id not in user_map:
        return []

    u_idx = user_map[user_id]
    scores = global_mean + user_bias[u_idx] + movie_bias + np.dot(Q, P[u_idx])
    reverse_movie_map = {v: k for k, v in movie_map.items()}
    watched_movies = train_df[train_df['user_id'] == user_id]['movie_id'].values

    for mid in watched_movies:
        if mid in movie_map:
            scores[movie_map[mid]] = -np.inf

    sorted_idx = np.argsort(scores)[::-1]
    return [reverse_movie_map[idx] for idx in sorted_idx[:N]]

def hit_rate_mf_sample(P, Q, user_bias, movie_bias, global_mean, user_map, movie_map, train_df, test_df, N=10, max_users=200):
    test_user_items = build_user_item_dict(test_df)
    all_users = list(test_user_items.keys())[:max_users]
    hits = 0
    total = 0
    for user_id in all_users:
        if user_id not in user_map:
            continue
        recommended = recommend_mf(P, Q, user_bias, movie_bias, global_mean, user_map, train_df, movie_map, user_id, N)
        actual = test_user_items[user_id]
        if any(item in actual for item in recommended):
            hits += 1
        total += 1
    return hits / total if total > 0 else 0

# === 顯示推薦結果 ===
def show_recommendations_mf(train_df, movie_titles_df, user_id, P, Q, user_bias, movie_bias, global_mean, user_map, movie_map, top_n=10):
    if user_id not in user_map:
        print(f"使用者 {user_id} 不在資料中")
        return
    recommendations = recommend_mf(P, Q, user_bias, movie_bias, global_mean, user_map, train_df, movie_map, user_id, top_n)
    print(f"\n Matrix Factorization 推薦給使用者 {user_id} 的前 {top_n} 部電影：")
    for idx, movie_id in enumerate(recommendations, 1):
        movie_title = movie_titles_df.loc[movie_titles_df['movie_id'] == movie_id, 'title'].values
        print(f"{idx}. {movie_title[0] if len(movie_title) > 0 else movie_id}")

# === 測試執行 ===
if __name__ == '__main__':
    # 測試資料載入
    df = load_netflix_data(data_dir="netflix", max_files=2, max_rows=500000)
    print(df.head())
    print(f"總筆數：{len(df)}，使用者數：{df['user_id'].nunique()}，電影數：{df['movie_id'].nunique()}")

    df = filter_sparse(df, min_movies=10, min_users=10)
    print(f"sparse 總筆數：{len(df)}，使用者數：{df['user_id'].nunique()}，電影數：{df['movie_id'].nunique()}")

    train_df, test_df = train_test_split_strict(df)

    P, Q, user_bias, movie_bias, global_mean, user_map, movie_map = matrix_factorization(train_df)
    rmse = evaluate_rmse_mf(test_df, P, Q, user_bias, movie_bias, global_mean, user_map, movie_map)
    print(f"\nFinal Test RMSE: {rmse:.4f}")

    hr = hit_rate_mf_sample(P, Q, user_bias, movie_bias, global_mean, user_map, movie_map, train_df, test_df)
    print(f"Top-10 Hit Rate: {hr:.4f}")

