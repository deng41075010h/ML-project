import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from collections import defaultdict

def build_user_item_matrix(df):
    user_item = defaultdict(dict)
    for _, row in df.iterrows():
        user_item[row['user_id']][row['movie_id']] = row['rating']
    return user_item

def predict_cf(user_item, user_id, movie_id, k=5):
    if user_id not in user_item:
        return 3.0  # default
    sims = []
    target_ratings = user_item[user_id]
    for other_id, other_ratings in user_item.items():
        if other_id == user_id or movie_id not in other_ratings:
            continue
        common = set(target_ratings) & set(other_ratings)
        if not common:
            continue
        a = np.array([target_ratings[m] for m in common])
        b = np.array([other_ratings[m] for m in common])
        sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
        sims.append((sim, other_ratings[movie_id]))
    if not sims:
        return 3.0
    sims.sort(reverse=True)
    top = sims[:k]
    return np.dot([s for s, _ in top], [r for _, r in top]) / (sum(s for s, _ in top) + 1e-9)

# 建立 item-user rating 資料結構
def build_item_user_matrix(df):
    item_user = defaultdict(dict)
    for _, row in df.iterrows():
        item_user[row['movie_id']][row['user_id']] = row['rating']
    return item_user

# 計算兩部電影的相似度（使用餘弦相似度）
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

# 預測使用者 user_id 對 movie_id 的評分
def predict_cf_item_based(item_user, user_item, user_id, movie_id, k=5):
    if user_id not in user_item:
        return 3.0  # 沒有資料的使用者給 default

    rated_items = user_item[user_id]
    sims = []

    for other_movie_id, rating in rated_items.items():
        if other_movie_id == movie_id:
            continue
        # 兩部電影的共同評分使用者
        users_i = item_user.get(movie_id, {})
        users_j = item_user.get(other_movie_id, {})
        common_users = set(users_i.keys()) & set(users_j.keys())

        if not common_users:
            continue

        vec_i = [users_i[u] for u in common_users]
        vec_j = [users_j[u] for u in common_users]
        sim = cosine_similarity(vec_i, vec_j)
        sims.append((sim, rating))

    if not sims:
        return 3.0  # 無法計算相似度時，回傳預設值

    sims.sort(reverse=True)
    top_k = sims[:k]

    # 加權平均
    numerator = sum(sim * r for sim, r in top_k)
    denominator = sum(abs(sim) for sim, _ in top_k) + 1e-9
    return numerator / denominator