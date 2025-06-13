# 主程式共用部分 (main.py)
import os
import pandas as pd
import numpy as np
from DataProcessing import load_netflix_data, filter_sparse, train_test_split_strict, verify_train_test_overlap

from UserBasedCF import run_user_based_cf
from ItemBasedCF import run_item_based_cf
from MatrixFactorizationMain import run_mf
from CBF import run_content_based

if __name__ == '__main__':
    df = load_netflix_data(data_dir="netflix", max_files=2, max_rows=500000)
    print(df.head())
    print(f"number of total: {len(df)}, number of user: {df['user_id'].nunique()}, number of movie: {df['movie_id'].nunique()}")

    df = filter_sparse(df, min_movies=10, min_users=10)
    print(f"sparse number of total: {len(df)}, number of user: {df['user_id'].nunique()}, number of movie: {df['movie_id'].nunique()}")

    train_df, test_df = train_test_split_strict(df)
    verify_train_test_overlap(train_df, test_df)

    print("\n--- Content Based Filtering ---")
    run_content_based(train_df, test_df)

    print("\n--- User Based CF ---")
    run_user_based_cf(train_df, test_df)

    print("\n--- Item Based CF ---")
    run_item_based_cf(train_df, test_df)

    print("\n--- Matrix Factorization ---")
    run_mf(train_df, test_df)
