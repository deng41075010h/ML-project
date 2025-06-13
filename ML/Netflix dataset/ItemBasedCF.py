from CollaborativeFiltering import build_user_item_matrix, build_item_user_matrix
from Evaluate import evaluate_rmse_cf_item_based, evaluate_metrics_cf_item
from HitRate import hit_rate_item_cf_sample, show_recommendations_item_cf
import pandas as pd

def run_item_based_cf(train_df, test_df):
    user_item = build_user_item_matrix(train_df)
    item_user = build_item_user_matrix(train_df)
    rmse = evaluate_rmse_cf_item_based(user_item, item_user, test_df, k=5)
    print(f"RMSE of Item-based CF = {rmse:.4f}")

    mse_item, rss_item, tss_item, r2_item = evaluate_metrics_cf_item(test_df, user_item, item_user)
    print(f"MSE of Item-based CF = {mse_item:.4f}\n"
          f"RSS of Item-based CF = {rss_item:.4f}\n"
          f"TSS of Item-based CF = {tss_item:.4f}\n"
          f"R2 of Item-based CF = {r2_item:.4f}")

    hr_cf = hit_rate_item_cf_sample(item_user, user_item, test_df, N=10, max_users=100)
    print(f"Item-based CF sampling Top-10 hit rate (100 people): {hr_cf:.4f}")

    # 顯示推薦清單
    movie_titles_df = load_movie_titles()
    test_user = test_df['user_id'].iloc[0]
    all_items = train_df['movie_id'].unique()
    show_recommendations_item_cf(movie_titles_df, test_user, item_user, user_item, all_items, top_n=10)

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
