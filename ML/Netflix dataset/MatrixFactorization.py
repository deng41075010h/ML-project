import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from collections import defaultdict

import numpy as np
from sklearn.metrics import mean_squared_error

def matrix_factorization(train, n_factors=20, n_iters=15, lr=0.008, reg=0.01):
    user_ids = train['user_id'].unique()
    movie_ids = train['movie_id'].unique()
    user_map = {u: i for i, u in enumerate(user_ids)}
    movie_map = {m: i for i, m in enumerate(movie_ids)}

    n_users, n_movies = len(user_ids), len(movie_ids)
    P = np.random.normal(0, 0.1, (n_users, n_factors))
    Q = np.random.normal(0, 0.1, (n_movies, n_factors))

    rmse_history = []

    for iter_num in range(1, n_iters + 1):
        for _, row in train.iterrows():
            u = user_map[row['user_id']]
            m = movie_map[row['movie_id']]
            r = row['rating']
            pred = np.dot(P[u], Q[m])
            err = r - pred
            P[u] += lr * (err * Q[m] - reg * P[u])
            Q[m] += lr * (err * P[u] - reg * Q[m])
        
        # 每次迭代後計算 RMSE
        preds = []
        truths = []
        
        for _, row in train.iterrows():
            u = user_map[row['user_id']]
            m = movie_map[row['movie_id']]
            pred = np.dot(P[u], Q[m])
            preds.append(pred)
            truths.append(row['rating'])

        rmse = np.sqrt(mean_squared_error(truths, preds))
        rmse_history.append(rmse)
        print(f"Iteration {iter_num}/{n_iters}, RMSE = {rmse:.4f}")

    # plotting RMSE history
    import matplotlib.pyplot as plt
    plt.plot(range(1, len(rmse_history)+1), rmse_history)
    plt.xlabel("Iteration")
    plt.ylabel("Training RMSE")
    plt.title("MF Convergence Curve")
    plt.grid(True)
    plt.show()


    return P, Q, user_map, movie_map
