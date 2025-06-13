from MatrixFactorization_cv import matrix_factorization
from Evaluate_cv import evaluate_rmse_mf, evaluate_metrics_mf
from HitRate_cv import hit_rate_mf_sample, show_recommendations_mf
import pandas as pd

def run_mf(train_df, test_df):
    P, Q, user_bias, movie_bias, global_mean, user_map, movie_map = matrix_factorization(train_df)
    rmse_mf = evaluate_rmse_mf(test_df, P, Q, user_bias, movie_bias, global_mean, user_map, movie_map)
    print(f"MF 的 RMSE = {rmse_mf:.4f}")

    mse_mf, rss_mf, tss_mf, r2_mf = evaluate_metrics_mf(test_df, P, Q, user_map, movie_map)
    print(f"MF 的 MSE = {mse_mf:.4f}\n"
          f"MF 的 RSS = {rss_mf:.4f}\n"
          f"MF 的 TSS = {tss_mf:.4f}\n"
          f"MF 的 R2 = {r2_mf:.4f}")

    hr_mf = hit_rate_mf_sample(P, Q, user_bias, movie_bias, global_mean, user_map, movie_map, train_df, test_df, 10, 100)
    print(f"MF 抽樣 Top-10 命中率(100人): {hr_mf:.4f}")

    movie_titles_df = load_movie_titles()
    test_user = test_df['user_id'].iloc[0]
    show_recommendations_mf(train_df, movie_titles_df, test_user, P, Q, user_bias, movie_bias, global_mean, user_map, movie_map, 10)

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
