# 主程式共用部分 (main.py)
import os
import pandas as pd
import numpy as np
from DataProcessing import load_netflix_data, filter_sparse, train_test_split_strict, verify_train_test_overlap
from Evaluate_cv import evaluate_rmse_cf, evaluate_rmse_mf, evaluate_rmse_cf_item_based
from HitRate_cv import hit_rate_cf_sample, hit_rate_mf_sample, show_recommendations_cf, show_recommendations_mf
import matplotlib.pyplot as plt

from UserBasedCF import run_user_based_cf
from ItemBasedCF import run_item_based_cf
from MatrixFactorizationMain_cv import run_mf
from cbf_2_with_decision_tree import run_content_based


if __name__ == '__main__':
    df = load_netflix_data(data_dir="netflix", max_files=2, max_rows=500000)
    print(df.head())
    print(f"總筆數：{len(df)}，使用者數：{df['user_id'].nunique()}，電影數：{df['movie_id'].nunique()}")

    df = filter_sparse(df, min_movies=10, min_users=10)
    print(f"sparse 總筆數：{len(df)}，使用者數：{df['user_id'].nunique()}，電影數：{df['movie_id'].nunique()}")

    train_df, test_df = train_test_split_strict(df)
    verify_train_test_overlap(train_df, test_df)

    # 指定 TMDB 路徑（你需將檔案放在 datasets 資料夾中）
    movies_path = "TMDB 5000 Movie Dataset/tmdb_5000_movies.csv"
    credits_path = "TMDB 5000 Movie Dataset/tmdb_5000_credits.csv"

    print("\n--- Content Based Filtering ---")
    # common_movies = train_df['movie_id'].isin(movie_df['movie_id']).sum()
    # print(f"Train set 中與 TMDB 資料匹配的 movie_id 數量：{common_movies}")

    run_content_based(train_df, test_df, movies_path, credits_path)

    print("\n--- User Based CF ---")
    run_user_based_cf(train_df, test_df)

    print("\n--- Item Based CF ---")
    run_item_based_cf(train_df, test_df)

    print("\n--- 矩陣分解 ---")
    run_mf(train_df, test_df)
