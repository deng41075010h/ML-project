import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from CollaborativeFiltering_cv import build_user_item_matrix
from Evaluate_cv import evaluate_rmse_cf, evaluate_metrics_cf_user
from HitRate_cv import hit_rate_cf_sample, show_recommendations_cf
from DataProcessing import load_netflix_data, filter_sparse  

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

def mean_center_user_ratings(user_item):
    centered = {}
    user_means = {}
    for uid, ratings in user_item.items():
        mu = np.mean(list(ratings.values()))
        user_means[uid] = mu
        centered[uid] = {item: rating - mu for item, rating in ratings.items()}
    return centered, user_means

def predict_cf_topk(user_item, target_user, target_item, K=30):
    from numpy.linalg import norm

    if target_user not in user_item:
        return 3.0  # 預設分數，避免 KeyError

    def cosine_sim(u1, u2):
        common = set(u1) & set(u2)
        if not common:
            return 0
        v1 = np.array([u1[i] for i in common])
        v2 = np.array([u2[i] for i in common])
        return np.dot(v1, v2) / (norm(v1) * norm(v2)) if norm(v1) and norm(v2) else 0

    similarities = []
    for other_user, ratings in user_item.items():
        if other_user == target_user:
            continue
        if target_item not in ratings:
            continue
        sim = cosine_sim(user_item[target_user], ratings)
        if sim > 0:
            similarities.append((sim, ratings[target_item]))

    similarities.sort(reverse=True)
    topk = similarities[:K]

    if not topk:
        return 0.0

    numerator = sum(sim * rating for sim, rating in topk)
    denominator = sum(abs(sim) for sim, _ in topk)
    return numerator / denominator if denominator != 0 else 0.0

def cross_validate_user_cf(ratings_df, k_folds=5):
    users = ratings_df['user_id'].unique()
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    rmse_scores, mse_scores, rss_scores, tss_scores, r2_scores, hr_scores = [], [], [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(users), 1):
        print(f"\nUser-Based Fold {fold}")
        test_users = users[test_idx]

        train_df = ratings_df[~ratings_df['user_id'].isin(test_users)]
        test_df = ratings_df[ratings_df['user_id'].isin(test_users)]

        user_item_df = build_user_item_matrix(train_df)

        # 強制轉換為 dict-of-dict 結構
        user_item = {}
        for user_id in user_item_df.index:
            user_item[int(user_id)] = {}
            for movie_id in user_item_df.columns:
                rating = user_item_df.at[user_id, movie_id]
                if not np.isnan(rating):
                    user_item[int(user_id)][int(movie_id)] = rating

        centered_user_item, user_means = mean_center_user_ratings(user_item)

        def predict(u, i):
            return predict_cf_topk(centered_user_item, int(u), int(i), K=30) + user_means.get(int(u), 0)

        # 預測與評估
        preds = [predict(row['user_id'], row['movie_id']) for _, row in test_df.iterrows()]
        actuals = test_df['rating'].values

        from sklearn.metrics import mean_squared_error, r2_score
        rmse = np.sqrt(mean_squared_error(actuals, preds))
        rmse_scores.append(rmse)
        print(f"RMSE = {rmse:.4f}")

        mse = mean_squared_error(actuals, preds)
        rss = np.sum((actuals - preds) ** 2)
        tss = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = r2_score(actuals, preds)

        mse_scores.append(mse)
        rss_scores.append(rss)
        tss_scores.append(tss)
        r2_scores.append(r2)
        print(f"MSE = {mse:.4f}, RSS = {rss:.4f}, TSS = {tss:.4f}, R2 = {r2:.4f}")

        # 命中率（手動改寫 hit_rate）
        test_actual = test_df.groupby('user_id')['movie_id'].apply(set).to_dict()
        all_items = train_df['movie_id'].unique()
        selected_users = list(test_actual.keys())[:100]
        hits, total = 0, 0

        for user in selected_users:
            watched = user_item.get(user, set())
            candidates = [item for item in all_items if item not in watched]
            scored = [(item, predict(user, item)) for item in candidates]
            scored.sort(key=lambda x: x[1], reverse=True)
            top_n = [item for item, _ in scored[:10]]
            actual = test_actual.get(user, set())
            if any(item in actual for item in top_n):
                hits += 1
            total += 1

        hr = hits / total if total > 0 else 0
        hr_scores.append(hr)
        print(f"Hit Rate@10 = {hr:.4f}")

    print("\nUser-Based Average Score: ")
    print(f"Average RMSE = {sum(rmse_scores)/k_folds:.4f}")
    print(f"Average MSE = {sum(mse_scores)/k_folds:.4f}")
    print(f"Average RSS = {sum(rss_scores)/k_folds:.4f}")
    print(f"Average TSS = {sum(tss_scores)/k_folds:.4f}")
    print(f"Average R² = {sum(r2_scores)/k_folds:.4f}")
    print(f"Average Hit Rate@10 = {sum(hr_scores)/k_folds:.4f}")



if __name__ == "__main__":
    # 從 Netflix 原始資料轉換而來
    df = load_netflix_data(
        data_dir="netflix",
        max_files=2,
        max_rows=500000
    )
    print(df.head())
    print(f"總筆數：{len(df)}，使用者數：{df['user_id'].nunique()}，電影數：{df['movie_id'].nunique()}")

    df = filter_sparse(df, min_movies=10, min_users=10)
    print(f"sparse 總筆數：{len(df)}，使用者數：{df['user_id'].nunique()}，電影數：{df['movie_id'].nunique()}")

    cross_validate_user_cf(df)

   
    



