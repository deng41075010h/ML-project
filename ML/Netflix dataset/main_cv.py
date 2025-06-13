# 主程式使用交叉驗證版本
import os
import pandas as pd
import numpy as np
from DataProcessing import load_netflix_data, filter_sparse, train_test_split_strict, verify_train_test_overlap
from Evaluate_cv import evaluate_rmse_cf, evaluate_rmse_mf, evaluate_rmse_cf_item_based
from HitRate_cv import hit_rate_cf_sample, hit_rate_mf_sample, show_recommendations_cf, show_recommendations_mf
import matplotlib.pyplot as plt

from MatrixFactorizationMain_cv import run_mf
from CBF import run_content_based
from UserBasedCF_cv import cross_validate_user_cf
from ItemBasedCF_cv import cross_validate_item_cf

if __name__ == '__main__':
    df = load_netflix_data(data_dir="netflix", max_files=2, max_rows=500000)
    print(df.head())
    print(f"number of total: {len(df)}, number of user: {df['user_id'].nunique()}, number of movie: {df['movie_id'].nunique()}")

    df = filter_sparse(df, min_movies=10, min_users=10)
    print(f"number of total: {len(df)}, number of user: {df['user_id'].nunique()}, number of movie: {df['movie_id'].nunique()}")

    train_df, test_df = train_test_split_strict(df)
    verify_train_test_overlap(train_df, test_df)



    print("\n--- User Based CF (Cross-Validation) ---")
    cross_validate_user_cf(df)



    print("\n--- Item Based CF (Cross-Validation) ---")
    cross_validate_item_cf(df)

