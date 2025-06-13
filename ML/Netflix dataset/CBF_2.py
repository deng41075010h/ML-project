import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

# === 載入電影資料（含 cast, crew, genres, keywords, overview） ===
def load_movie_data(movies_path, credits_path):
    movies_df = pd.read_csv(movies_path)
    credits_df = pd.read_csv(credits_path)
    df = movies_df.merge(credits_df, left_on='id', right_on='movie_id')
    df['title'] = df['title_x']
    df['director'] = df['crew'].apply(extract_director)
    df['cast_names'] = df['cast'].apply(extract_cast)
    df['genres_text'] = df['genres'].apply(extract_keywords)
    df['keywords_text'] = df['keywords'].apply(extract_keywords)
    df['content'] = (
        df['overview'].fillna('') + ' ' +
        df['genres_text'] + ' ' +
        df['keywords_text'] + ' ' +
        (df['cast_names'] + ' ') * 2 +
        (df['director'].fillna('') + ' ') * 3
    )

    #  篩掉內容太短的（例如少於 20 個字）
    df = df[df['content'].str.len() > 20]

    return df[['id', 'title', 'content']].rename(columns={'id': 'movie_id'})

def extract_director(crew_json):
    try:
        crew_list = ast.literal_eval(crew_json)
        for person in crew_list:
            if person.get('job') == 'Director':
                return person.get('name')
    except:
        return None

def extract_cast(cast_json):
    try:
        cast_list = ast.literal_eval(cast_json)
        return ' '.join([person['name'] for person in cast_list[:3]])
    except:
        return ''

def extract_keywords(json_str):
    try:
        items = ast.literal_eval(json_str)
        return ' '.join([item['name'] for item in items])
    except:
        return ''

# === 建立 TF-IDF 模型 ===
def build_tfidf_model(movie_df):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(movie_df['content'])
    return vectorizer, tfidf_matrix

# === 根據使用者喜好推薦電影 ===
def recommend_cbf_for_user(user_id, train_df, movie_df, tfidf_matrix, top_n=10):
    liked_movies = train_df[(train_df['user_id'] == user_id) & (train_df['rating'] >= 4)]
    liked_movies = liked_movies[liked_movies['movie_id'].isin(movie_df['movie_id'])]  # 確保出現在 movie_df 中
    liked_ids = liked_movies['movie_id'].values

    if len(liked_ids) == 0:
        return []

    liked_indices = movie_df[movie_df['movie_id'].isin(liked_ids)].index
    weights = liked_movies.set_index('movie_id').loc[movie_df.loc[liked_indices]['movie_id']]['rating'].values

    # 若長度仍對不上，防守性檢查
    if len(weights) != len(liked_indices):
        return []

    user_profile = np.average(tfidf_matrix[liked_indices].toarray(), axis=0, weights=weights)
    cos_sim = cosine_similarity([user_profile], tfidf_matrix).flatten()

    # 排除已看過
    seen_indices = movie_df[movie_df['movie_id'].isin(liked_ids)].index
    cos_sim[seen_indices] = -1

    #  可選：過濾相似度過低的項目（例如 < 0.05）
    cos_sim[cos_sim < 0.05] = -1

    top_indices = cos_sim.argsort()[::-1][:top_n]
    recommended_ids = movie_df.iloc[top_indices]['movie_id'].tolist()
    return recommended_ids

def show_recommendations_cbf(movie_df, train_df, user_id, tfidf_matrix, top_n=10):
    recs = recommend_cbf_for_user(user_id, train_df, movie_df, tfidf_matrix, top_n=top_n)
    titles = movie_df[movie_df['movie_id'].isin(recs)]['title'].tolist()
    print(f"\n推薦給使用者 {user_id} 的前 {top_n} 部電影：")
    for i, title in enumerate(titles, 1):
        print(f"{i}. {title}")

# === 評估：命中率與 RMSE ===
def hit_rate_cbf_sample(train_df, test_df, movie_df, tfidf_matrix, N=10, max_users=100):
    sampled_users = test_df['user_id'].drop_duplicates().sample(n=min(max_users, test_df['user_id'].nunique()), random_state=42)
    hits = 0
    total = 0
    for user_id in sampled_users:
        test_movies = test_df[test_df['user_id'] == user_id]['movie_id'].tolist()
        recs = recommend_cbf_for_user(user_id, train_df, movie_df, tfidf_matrix, top_n=N)
        if any(movie in recs for movie in test_movies):
            hits += 1
        total += 1
    return hits / total if total > 0 else 0

def predict_rating_cbf(user_id, movie_id, train_df, movie_df, tfidf_matrix):
    # 取出該使用者喜歡的電影（評分 >= 4）
    liked_movies = train_df[(train_df['user_id'] == user_id) & (train_df['rating'] >= 4)]
    liked_ids = liked_movies['movie_id'].values

    #  防呆 1：使用者沒有喜歡的電影
    if len(liked_ids) == 0:
        return 3.0

    # 取得 target movie 在 tfidf 矩陣中的位置
    movie_idx_list = movie_df[movie_df['movie_id'] == movie_id].index
    if len(movie_idx_list) == 0:
        return 3.0
    movie_idx = movie_idx_list[0]
    target_vec = tfidf_matrix[movie_idx]

    # 取得使用者喜歡的電影在 tfidf 中的位置
    liked_indices = movie_df[movie_df['movie_id'].isin(liked_ids)].index

    #  防呆 2：使用者喜歡的電影不在 TF-IDF movie_df 裡
    if len(liked_indices) == 0:
        return 3.0

    liked_vecs = tfidf_matrix[liked_indices]

    #  防呆 3：再確認 liked_vecs 不是空矩陣
    if liked_vecs.shape[0] == 0:
        return 3.0

    # 計算 cosine similarity
    sims = cosine_similarity(target_vec, liked_vecs).flatten()

    liked_movie_ids = movie_df.loc[liked_indices]['movie_id']
    try:
        liked_ratings = liked_movies.set_index('movie_id').loc[liked_movie_ids]['rating'].values
    except KeyError:
        return 3.0

    if sims.sum() == 0:
        return 3.0

    pred_rating = np.dot(sims, liked_ratings) / sims.sum()
    return pred_rating



def compute_rmse_cbf(train_df, test_df, movie_df, tfidf_matrix):
    y_true = []
    y_pred = []
    for _, row in test_df.iterrows():
        user_id = row['user_id']
        movie_id = row['movie_id']
        true_rating = row['rating']
        pred_rating = predict_rating_cbf(user_id, movie_id, train_df, movie_df, tfidf_matrix)
        y_true.append(true_rating)
        y_pred.append(pred_rating)
    return np.sqrt(mean_squared_error(y_true, y_pred))

def run_content_based(train_df, test_df, movies_path, credits_path):
    # 載入電影資料（含 content 欄）
    movie_df = load_movie_data(movies_path, credits_path)

    # 建立 TF-IDF 模型
    vectorizer, tfidf_matrix = build_tfidf_model(movie_df)

    # 計算並印出 RMSE
    rmse_cbf = compute_rmse_cbf(train_df, test_df, movie_df, tfidf_matrix)
    print(f"CBF 的 RMSE : {rmse_cbf:.4f}")

    # 評估抽樣 Top-N 命中率
    hr_cbf = hit_rate_cbf_sample(train_df, test_df, movie_df, tfidf_matrix, N=50, max_users=100)
    print(f"CBF 抽樣 Top-50 命中率(100人): {hr_cbf:.4f}")

    # 顯示推薦
    test_user = test_df['user_id'].iloc[0]
    show_recommendations_cbf(movie_df, train_df, test_user, tfidf_matrix, top_n=10)
