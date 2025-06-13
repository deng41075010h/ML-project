import pandas as pd
from sklearn.model_selection import KFold
from CollaborativeFiltering_cv import build_user_item_matrix, build_item_user_matrix
from Evaluate_cv import evaluate_rmse_cf_item_based, evaluate_metrics_cf_item
from HitRate_cv import hit_rate_cf_item_sample, show_recommendations_cf_item_based

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

def cross_validate_item_cf(ratings_df, k_folds=5):
    users = ratings_df['user_id'].unique()
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    rmse_scores, mse_scores, rss_scores, tss_scores, r2_scores, hr_scores = [], [], [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(users), 1):
        print(f"\nItem-Based Fold {fold}")
        test_users = users[test_idx]

        train_df = ratings_df[~ratings_df['user_id'].isin(test_users)]
        test_df = ratings_df[ratings_df['user_id'].isin(test_users)]

        user_item = build_user_item_matrix(train_df)
        item_user = build_item_user_matrix(train_df)

        rmse = evaluate_rmse_cf_item_based(user_item, item_user, test_df, k=5)
        rmse_scores.append(rmse)
        print(f"RMSE = {rmse:.4f}")

        mse, rss, tss, r2 = evaluate_metrics_cf_item(test_df, user_item, item_user, k=5)
        mse_scores.append(mse)
        rss_scores.append(rss)
        tss_scores.append(tss)
        r2_scores.append(r2)
        print(f"MSE = {mse:.4f}, RSS = {rss:.4f}, TSS = {tss:.4f}, R2 = {r2:.4f}")



    print("\nItem-Based Average Score:")
    print(f"Average RMSE = {sum(rmse_scores)/k_folds:.4f}")
    print(f"Average MSE = {sum(mse_scores)/k_folds:.4f}")
    print(f"Average RSS = {sum(rss_scores)/k_folds:.4f}")
    print(f"Average TSS = {sum(tss_scores)/k_folds:.4f}")
    print(f"Average R² = {sum(r2_scores)/k_folds:.4f}")
    

if __name__ == "__main__":
    ratings_df = pd.read_csv("netflix/ratings.csv")  # 請換成實際路徑
    print("==== Item-based Collaborative Filtering with Cross-Validation ====")
    cross_validate_item_cf(ratings_df)
