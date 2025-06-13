import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from collections import defaultdict
from CollaborativeFiltering import build_user_item_matrix, predict_cf, predict_cf_item_based

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

# --- Collaborative Filtering 的 Top-N 推薦與命中率 ---
# user based CF
def recommend_cf(user_item, target_user, all_items, N=10):
    watched = user_item.get(target_user, set())
    candidates = [item for item in all_items if item not in watched]
    scored = [(item, predict_cf(user_item, target_user, item)) for item in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [item for item, _ in scored[:N]]

# item based CF
def recommend_item_based_cf(item_user, user_item, user_id, all_items, N=10):
    watched = set(user_item.get(user_id, {}).keys())
    candidates = [item for item in all_items if item not in watched]
    
    scored = [(item, predict_cf_item_based(item_user, user_item, user_id, item)) for item in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    
    return [item for item, _ in scored[:N]]




# user based CF 的抽樣 hit rate
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

# item based CF 的抽樣 hit rate
def hit_rate_item_cf_sample(item_user, user_item, test_df, N=10, max_users=100):
    all_items = test_df['movie_id'].unique()
    test_actual = build_user_item_dict(test_df)
    selected_users = list(test_actual.keys())[:max_users]

    hits = 0
    total = 0
    for user in selected_users:
        if user not in user_item:
            continue
        recs = recommend_item_based_cf(item_user, user_item, user, all_items, N)
        actual = test_actual[user]
        if any(item in actual for item in recs):
            hits += 1
        total += 1
    return hits / total if total > 0 else 0


# --- Matrix Factorization 的 Top-N 推薦與命中率 ---
def recommend_mf(P, Q, user_map, train_df, movie_map, user_id, N=10):
    if user_id not in user_map:
        return []

    u_idx = user_map[user_id]
    scores = np.dot(Q, P[u_idx])

    reverse_movie_map = {v: k for k, v in movie_map.items()}
    watched_movies = train_df[train_df['user_id'] == user_id]['movie_id'].values

    # 將已看過的電影的分數設為 -inf，避免推薦
    for mid in watched_movies:
        if mid in movie_map:
            scores[movie_map[mid]] = -np.inf

    sorted_idx = np.argsort(scores)[::-1]
    recommendations = []
    for idx in sorted_idx:
        movie_id = reverse_movie_map[idx]
        recommendations.append(movie_id)
        if len(recommendations) == N:
            break
    return recommendations


def hit_rate_mf(P, Q, user_map, movie_map, train_df, test_df, N=10):
    global test_user_items
    test_user_items = build_user_item_dict(test_df)
    all_users = test_df['user_id'].unique()
    hits = 0
    total = 0
    for user_id in all_users:
        if user_id not in user_map:
            continue
        recommended = recommend_mf(P, Q, user_map, train_df, movie_map, user_id, N)
        actual = test_user_items[user_id]
        if any(item in actual for item in recommended):
            hits += 1
        total += 1
    return hits / total if total > 0 else 0

def hit_rate_mf_sample(P, Q, user_map, movie_map, train_df, test_df, N=10, max_users=200):
    global test_user_items
    test_user_items = build_user_item_dict(test_df)
    all_users = list(test_user_items.keys())[:max_users]

    hits = 0
    total = 0
    for user_id in all_users:
        if user_id not in user_map:
            continue
        recommended = recommend_mf(P, Q, user_map, train_df, movie_map, user_id, N)
        actual = test_user_items[user_id]
        if any(item in actual for item in recommended):
            hits += 1
        total += 1
    return hits / total if total > 0 else 0


# --- 顯示推薦結果 ---
# user based CF
def show_recommendations_cf(movie_titles_df, user_id, user_item, all_items, top_n=10):
    if user_id not in user_item:
        print(f"user {user_id} does not exist in the profile")
        return
    watched = user_item[user_id]
    recommendations = recommend_cf(user_item, user_id, all_items, top_n)
    print(f"\n Collaborative Filtering recommends the top {top_n} movies to user {user_id} :")
    for idx, movie_id in enumerate(recommendations, 1):
        movie_title = movie_titles_df.loc[movie_titles_df['movie_id'] == movie_id, 'title'].values
        print(f"{idx}. {movie_title[0] if len(movie_title) > 0 else movie_id}")

# item based CF
def show_recommendations_item_cf(movie_titles_df, user_id, item_user, user_item, all_items, top_n=10):
    if user_id not in user_item:
        print(f"user {user_id} does not exist in the profile")
        return
    recs = recommend_item_based_cf(item_user, user_item, user_id, all_items, top_n)
    print(f"\nItem-Based CF recommends the top {top_n} movies to user {user_id} :")
    for i, movie_id in enumerate(recs, 1):
        title = movie_titles_df.loc[movie_titles_df['movie_id'] == movie_id, 'title'].values
        print(f"{i}. {title[0] if len(title) > 0 else movie_id}")

# Matrix Factorization
def show_recommendations_mf(train_df, movie_titles_df, user_id, P, Q, user_map, movie_map, top_n=10):
    if user_id not in user_map:
        print(f"user {user_id} does not exist in the profile")
        return
    recommendations = recommend_mf(P, Q, user_map, train_df, movie_map, user_id, top_n)
    print(f"\n Matrix Factorization recommends the top {top_n} movies to user {user_id} :")
    for idx, movie_id in enumerate(recommendations, 1):
        movie_title = movie_titles_df.loc[movie_titles_df['movie_id'] == movie_id, 'title'].values
        print(f"{idx}. {movie_title[0] if len(movie_title) > 0 else movie_id}")

