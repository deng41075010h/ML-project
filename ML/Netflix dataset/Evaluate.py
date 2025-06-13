import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from collections import defaultdict
from CollaborativeFiltering import build_user_item_matrix, predict_cf, predict_cf_item_based

# --- 評估 RMSE ---
def evaluate_rmse_cf(user_item, test_df):
    preds = [predict_cf(user_item, row['user_id'], row['movie_id']) for _, row in test_df.iterrows()]
    rmse = np.sqrt(mean_squared_error(test_df['rating'], preds))
    return rmse

def evaluate_rmse_cf_item_based(user_item, item_user, test_df, k=5):
    preds = [
        predict_cf_item_based(item_user, user_item, row['user_id'], row['movie_id'], k)
        for _, row in test_df.iterrows()
    ]
    rmse = np.sqrt(mean_squared_error(test_df['rating'], preds))
    return rmse

def evaluate_rmse_mf(test_df, P, Q, user_map, movie_map):
    preds = []
    actuals = []
    for _, row in test_df.iterrows():
        uid, mid = row['user_id'], row['movie_id']
        if uid in user_map and mid in movie_map:
            u, m = user_map[uid], movie_map[mid]
            pred = np.dot(P[u], Q[m])
            preds.append(pred)
            actuals.append(row['rating'])
    return np.sqrt(mean_squared_error(actuals, preds))

# --- 其他評估指標 ---
# User-based CF

def evaluate_metrics_cf_user(test_df, user_item):
    preds = [predict_cf(user_item, row['user_id'], row['movie_id']) for _, row in test_df.iterrows()]
    actuals = test_df['rating'].values

    mse = mean_squared_error(actuals, preds)
    rss = np.sum((actuals - preds) ** 2)
    tss = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = r2_score(actuals, preds)

    return mse,rss,tss,r2
    

# Item-based CF

def evaluate_metrics_cf_item(test_df, user_item, item_user, k=5):
    preds = [
        predict_cf_item_based(item_user, user_item, row['user_id'], row['movie_id'], k)
        for _, row in test_df.iterrows()
    ]
    actuals = test_df['rating'].values

    mse = mean_squared_error(actuals, preds)
    rss = np.sum((actuals - preds) ** 2)
    tss = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = r2_score(actuals, preds)

    return mse,rss,tss,r2
    

# Matrix Factorization

def evaluate_metrics_mf(test_df, P, Q, user_map, movie_map):
    preds = []
    actuals = []
    for _, row in test_df.iterrows():
        uid, mid = row['user_id'], row['movie_id']
        if uid in user_map and mid in movie_map:
            u, m = user_map[uid], movie_map[mid]
            pred = np.dot(P[u], Q[m])
            preds.append(pred)
            actuals.append(row['rating'])

    preds = np.array(preds)
    actuals = np.array(actuals)

    mse = mean_squared_error(actuals, preds)
    rss = np.sum((actuals - preds) ** 2)
    tss = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = r2_score(actuals, preds)

    return mse,rss,tss,r2
    
