import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from collections import defaultdict
import matplotlib.pyplot as plt

def load_netflix_data(
    data_dir="netflix",
    file_prefix="combined_data_",
    max_files=1,
    max_rows=None
):
    all_dfs = []
    file_count = 0
    total_rows = 0

    for i in range(1, 5):  # 檔案最多是 combined_data_1 ~ combined_data_4
        if file_count >= max_files:
            break
        file_path = os.path.join(data_dir, f"{file_prefix}{i}.txt")
        with open(file_path, 'r') as f:
            movie_id = None
            data = []
            for line in f:
                line = line.strip()
                if line.endswith(':'):
                    movie_id = int(line[:-1])
                else:
                    user_id, rating, date = line.split(',')
                    data.append((int(user_id), movie_id, int(rating)))
                    total_rows += 1
                    if max_rows and total_rows >= max_rows:
                        break
            df = pd.DataFrame(data, columns=['user_id', 'movie_id', 'rating'])
            all_dfs.append(df)
            file_count += 1
            if max_rows and total_rows >= max_rows:
                break
    return pd.concat(all_dfs, ignore_index=True)

def filter_sparse(df, min_movies=10, min_users=10):
    user_counts = df['user_id'].value_counts()
    movie_counts = df['movie_id'].value_counts()
    
    filtered_df = df[
        (df['user_id'].isin(user_counts[user_counts >= min_movies].index)) &
        (df['movie_id'].isin(movie_counts[movie_counts >= min_users].index))
    ]
    return filtered_df

def train_test_split_strict(df, test_size=0.2, random_state=42):
    # 先 shuffle
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    # 避免測試集中有新 user/movie
    unique_users = df['user_id'].unique()
    unique_movies = df['movie_id'].unique()
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    test = test[test['user_id'].isin(train['user_id']) & test['movie_id'].isin(train['movie_id'])]
    return train, test

def verify_train_test_overlap(train_df, test_df):
    train_users = set(train_df['user_id'])
    train_movies = set(train_df['movie_id'])

    test_users = set(test_df['user_id'])
    test_movies = set(test_df['movie_id'])

    user_coverage = len(test_users & train_users) / len(test_users) * 100
    movie_coverage = len(test_movies & train_movies) / len(test_movies) * 100

    print("\n-----------------------------\n")
    print("verify_train_test_overlap :\n")
    print(f"Test set user coverage: {user_coverage:.2f}%")
    print(f"Test set movie coverage: {movie_coverage:.2f}%")

    if user_coverage < 100 or movie_coverage < 100:
        print("Warning: There are users or movies in the test set that do not appear in the training set, which may cause the model to be unable to predict.")
    else:
        print("The test set is valid, all users and movies appear in the training set")