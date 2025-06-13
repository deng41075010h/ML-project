import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from collections import defaultdict
from CollaborativeFiltering_cv import build_user_item_matrix, predict_cf, predict_cf_item_based

# --- Top-N Recommendations 與 hit rate ---
def get_all_users_items(train_df):
    all_users = train_df['user_id'].unique()
    all_items = train_df['movie_id'].unique()
    return all_users, all_items

def build_user_item_dict(df):
    user_item = {}
    for _, row in df.iterrows():
        user_item.setdefault(row['user_id'], set()).add(row['movie_id'])
    return user_item

# --- User-based Collaborative Filtering 推薦與命中率 ---
def recommend_cf(user_item, target_user, all_items, N=10):
    watched = user_item.get(target_user, set())
    candidates = [item for item in all_items if item not in watched]
    scored = [(item, predict_cf(user_item, target_user, item)) for item in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [item for item, _ in scored[:N]]

def hit_rate_cf(user_item, test_df, N=10):
    all_users, all_items = get_all_users_items(test_df)
    test_actual = build_user_item_dict(test_df)

    hits = 0
    total = 0
    for user in test_actual:
        if user not in user_item:
            continue
        recommended = recommend_cf(user_item, user, all_items, N)
        actual = test_actual[user]
        if any(item in actual for item in recommended):
            hits += 1
        total += 1
    return hits / total if total > 0 else 0

# --- Item-based Collaborative Filtering 推薦 ---
def recommend_cf_item_based(user_item, item_user, target_user, all_items, N=10, k=5):
    watched = user_item.get(target_user, set())
    candidates = [item for item in all_items if item not in watched]
    scored = [
        (item, predict_cf_item_based(item_user, user_item, target_user, item, k))
        for item in candidates
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [item for item, _ in scored[:N]]

# --- User-based Hit Rate (樣本) ---
def hit_rate_cf_sample(user_item, test_df, N=10, max_users=200):
    all_users, all_items = get_all_users_items(test_df)
    test_actual = build_user_item_dict(test_df)
    selected_users = list(test_actual.keys())[:max_users]

    hits = 0
    total = 0
    for user in selected_users:
        if user not in user_item:
            continue
        recommended = recommend_cf(user_item, user, all_items, N)
        actual = test_actual[user]
        if any(item in actual for item in recommended):
            hits += 1
        total += 1
    return hits / total if total > 0 else 0

# --- Item-based Hit Rate (樣本) ---
def hit_rate_cf_item_sample(user_item, item_user, test_df, N=10, max_users=200, k=5):
    all_users, all_items = get_all_users_items(test_df)
    test_actual = build_user_item_dict(test_df)
    selected_users = list(test_actual.keys())[:max_users]

    hits = 0
    total = 0
    for user in selected_users:
        if user not in user_item:
            continue
        recommended = recommend_cf_item_based(user_item, item_user, user, all_items, N, k)
        actual = test_actual[user]
        if any(item in actual for item in recommended):
            hits += 1
        total += 1
    return hits / total if total > 0 else 0

# --- 顯示推薦結果 ---
def show_recommendations_cf(movie_titles_df, user_id, user_item, all_items, top_n=10):
    if user_id not in user_item:
        print(f"\u4f7f\u7528\u8005 {user_id} \u4e0d\u5728\u8cc7\u6599\u4e2d")
        return
    watched = user_item[user_id]
    recommendations = recommend_cf(user_item, user_id, all_items, top_n)
    print(f"\n Collaborative Filtering \u63a8\u85a6\u7d66\u4f7f\u7528\u8005 {user_id} \u7684\u524d {top_n} \u90e8\u96fb\u5f71:")
    for idx, movie_id in enumerate(recommendations, 1):
        movie_title = movie_titles_df.loc[movie_titles_df['movie_id'] == movie_id, 'title'].values
        print(f"{idx}. {movie_title[0] if len(movie_title) > 0 else movie_id}")

def show_recommendations_cf_item_based(movie_titles_df, user_id, user_item, item_user, all_items, top_n=10, k=5):
    if user_id not in user_item:
        print(f"\u4f7f\u7528\u8005 {user_id} \u4e0d\u5728\u8cc7\u6599\u4e2d")
        return
    watched = user_item[user_id]
    recommendations = recommend_cf_item_based(user_item, item_user, user_id, all_items, top_n, k)
    print(f"\nItem-based CF \u63a8\u85a6\u7d66\u4f7f\u7528\u8005 {user_id} \u7684\u524d {top_n} \u90e8\u96fb\u5f71:")
    for idx, movie_id in enumerate(recommendations, 1):
        movie_title = movie_titles_df.loc[movie_titles_df['movie_id'] == movie_id, 'title'].values
        print(f"{idx}. {movie_title[0] if len(movie_title) > 0 else movie_id}")

def show_recommendations_mf(train_df, movie_titles_df, user_id, P, Q, user_bias, movie_bias, global_mean, user_map, movie_map, top_n=10):
    if user_id not in user_map:
        print(f"\u4f7f\u7528\u8005 {user_id} \u4e0d\u5728\u8cc7\u6599\u4e2d")
        return

    u = user_map[user_id]
    scores = global_mean + user_bias[u] + movie_bias + np.dot(Q, P[u])

    watched = train_df[train_df['user_id'] == user_id]['movie_id'].values
    watched_indices = [movie_map[mid] for mid in watched if mid in movie_map]
    scores[watched_indices] = -np.inf

    reverse_map = {v: k for k, v in movie_map.items()}
    top_n_ids = [reverse_map[i] for i in np.argsort(scores)[::-1][:top_n]]
    top_n_titles = movie_titles_df[movie_titles_df['movie_id'].isin(top_n_ids)]['title'].tolist()

    print(f"\nMatrix Factorization \u63a8\u85a6\u7d66\u4f7f\u7528\u8005 {user_id} \u7684\u524d {top_n} \u90e8\u96fb\u5f71:")
    for i, title in enumerate(top_n_titles, 1):
        print(f"{i}. {title}")

def hit_rate_mf_sample(P, Q, user_bias, movie_bias, global_mean, user_map, movie_map, train_df, test_df, N=10, max_users=200):
    test_user_items = defaultdict(set)
    for _, row in test_df.iterrows():
        test_user_items[row['user_id']].add(row['movie_id'])

    all_users = list(test_user_items.keys())[:max_users]
    hits = 0
    total = 0

    for user_id in all_users:
        if user_id not in user_map:
            continue
        u = user_map[user_id]
        scores = global_mean + user_bias[u] + movie_bias + np.dot(Q, P[u])
        watched = train_df[train_df['user_id'] == user_id]['movie_id'].values
        watched_indices = [movie_map[mid] for mid in watched if mid in movie_map]
        scores[watched_indices] = -np.inf  # 排除已看過

        reverse_map = {v: k for k, v in movie_map.items()}
        top_n = [reverse_map[i] for i in np.argsort(scores)[::-1][:N]]
        if any(movie in test_user_items[user_id] for movie in top_n):
            hits += 1
        total += 1

    return hits / total if total > 0 else 0
