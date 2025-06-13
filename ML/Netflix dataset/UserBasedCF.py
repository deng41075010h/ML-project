from CollaborativeFiltering import build_user_item_matrix, predict_cf
from Evaluate import evaluate_rmse_cf, evaluate_metrics_cf_user
from HitRate import hit_rate_cf_sample, show_recommendations_cf
import pandas as pd

def run_user_based_cf(train_df, test_df):
    user_item = build_user_item_matrix(train_df)
    rmse = evaluate_rmse_cf(user_item, test_df)
    print(f"RMSE of User-based CF = {rmse:.4f}")

    mse_user, rss_user, tss_user, r2_user = evaluate_metrics_cf_user(test_df, user_item)
    print(f"MSE of User-based CF = {mse_user:.4f}\n"
          f"RSS of User-based CF = {rss_user:.4f}\n"
          f"TSS of User-based CF = {tss_user:.4f}\n"
          f"R2 of User-based CF = {r2_user:.4f}")

    hr_user_cf = hit_rate_cf_sample(user_item, test_df, N=10, max_users=100)
    print(f"User-based CF sampling Top-10 hit rate (100 people): {hr_user_cf:.4f}")

    # 顯示推薦清單
    movie_titles_df = load_movie_titles()
    test_user = test_df['user_id'].iloc[0]
    all_items = train_df['movie_id'].unique()
    show_recommendations_cf(movie_titles_df, test_user, user_item, all_items, top_n=10)

def load_movie_titles():
    corrected_data = []
    with open('netflix/movie_titles.csv', encoding='latin1') as f:
        for line in f:
            parts = line.strip().split(',')
            movie_id = int(parts[0])
            try:
                release_year = int(parts[1])
            except ValueError:
                release_year = None
            title = ','.join(parts[2:]).strip()
            corrected_data.append([movie_id, release_year, title])
    return pd.DataFrame(corrected_data, columns=['movie_id', 'release_year', 'title'])
