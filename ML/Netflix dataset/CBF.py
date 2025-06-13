import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

import sklearn.metrics
print(mean_squared_error.__module__)


def build_tfidf_model(movie_titles_df):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(movie_titles_df['title'])
    return vectorizer, tfidf_matrix

def recommend_cbf_for_user(user_id, train_df, movie_titles_df, tfidf_matrix, top_n=10):
    liked_movies = train_df[(train_df['user_id'] == user_id) & (train_df['rating'] >= 4)]
    liked_ids = liked_movies['movie_id'].unique()
    liked_indices = movie_titles_df[movie_titles_df['movie_id'].isin(liked_ids)].index

    if len(liked_indices) == 0:
        return []

    weights = liked_movies['rating'].values
    user_profile = np.average(tfidf_matrix[liked_indices].toarray(), axis=0, weights=weights)
    
    cos_sim = cosine_similarity([user_profile], tfidf_matrix).flatten()

    # 排除已看過電影
    seen_indices = movie_titles_df[movie_titles_df['movie_id'].isin(liked_ids)].index
    cos_sim[seen_indices] = -1

    top_indices = cos_sim.argsort()[::-1][:top_n]
    recommended_ids = movie_titles_df.iloc[top_indices]['movie_id'].tolist()
    return recommended_ids

def hit_rate_cbf_sample(train_df, test_df, movie_titles_df, tfidf_matrix, N=10, max_users=100):
    sampled_users = test_df['user_id'].drop_duplicates().sample(n=min(max_users, test_df['user_id'].nunique()), random_state=42)
    hits = 0
    total = 0

    for user_id in sampled_users:
        test_movies = test_df[test_df['user_id'] == user_id]['movie_id'].tolist()
        recs = recommend_cbf_for_user(user_id, train_df, movie_titles_df, tfidf_matrix, top_n=N)
        if any(movie in recs for movie in test_movies):
            hits += 1
        total += 1

    return hits / total if total > 0 else 0

def show_recommendations_cbf(movie_titles_df, train_df, user_id, tfidf_matrix, top_n=10):
    recs = recommend_cbf_for_user(user_id, train_df, movie_titles_df, tfidf_matrix, top_n=top_n)
    titles = movie_titles_df[movie_titles_df['movie_id'].isin(recs)]['title'].tolist()
    print(f"\nContent-Based Filtering recommends the top {top_n} movies to user {user_id} :")
    for i, title in enumerate(titles, 1):
        print(f"{i}. {title}")

# 新增的函式：用相似度加權平均計算預測評分
def predict_rating_cbf(user_id, movie_id, train_df, movie_titles_df, tfidf_matrix):
    liked_movies = train_df[(train_df['user_id'] == user_id) & (train_df['rating'] >= 4)]
    liked_ids = liked_movies['movie_id'].values

    if len(liked_ids) == 0:
        return 3.0  # 沒喜歡的電影，回傳中間值

    movie_idx = movie_titles_df[movie_titles_df['movie_id'] == movie_id].index
    if len(movie_idx) == 0:
        return 3.0  # 該電影不在資料中

    movie_idx = movie_idx[0]
    target_vec = tfidf_matrix[movie_idx]
    liked_indices = movie_titles_df[movie_titles_df['movie_id'].isin(liked_ids)].index
    liked_vecs = tfidf_matrix[liked_indices]

    sims = cosine_similarity(target_vec, liked_vecs).flatten()
    liked_ratings = liked_movies.set_index('movie_id').loc[movie_titles_df.loc[liked_indices]['movie_id']]['rating'].values

    if sims.sum() == 0:
        return 3.0

    pred_rating = np.dot(sims, liked_ratings) / sims.sum()
    return pred_rating



import sklearn.metrics as metrics
import numpy as np

def compute_rmse_cbf(train_df, test_df, movie_titles_df, tfidf_matrix):
    y_true = []
    y_pred = []

    for _, row in test_df.iterrows():
        user_id = row['user_id']
        movie_id = row['movie_id']
        true_rating = row['rating']
        pred_rating = predict_rating_cbf(user_id, movie_id, train_df, movie_titles_df, tfidf_matrix)
        y_true.append(true_rating)
        y_pred.append(pred_rating)

    # 用 np.sqrt 手動計算 RMSE，避免參數爭議
    rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
    return rmse


def run_content_based(train_df, test_df):
    movie_titles_df = load_movie_titles()
    vectorizer, tfidf_matrix = build_tfidf_model(movie_titles_df)


    # 計算並印出 RMSE
    rmse_cbf = compute_rmse_cbf(train_df, test_df, movie_titles_df, tfidf_matrix)
    print(f"RMSE of CBF : {rmse_cbf:.4f}")

    # 評估抽樣 Top-N 命中率
    hr_cbf = hit_rate_cbf_sample(train_df, test_df, movie_titles_df, tfidf_matrix, N=10, max_users=100)
    print(f"CBF sampling Top-10 hit rate (100 people): {hr_cbf:.4f}")


    # 顯示推薦
    test_user = test_df['user_id'].iloc[0]
    show_recommendations_cbf(movie_titles_df, train_df, test_user, tfidf_matrix, top_n=10)

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
