from MatrixFactorization import matrix_factorization
from Evaluate import evaluate_rmse_mf, evaluate_metrics_mf
from HitRate import hit_rate_mf_sample, show_recommendations_mf
import pandas as pd

def run_mf(train_df, test_df):
    P, Q, user_map, movie_map = matrix_factorization(train_df)
    rmse_mf = evaluate_rmse_mf(test_df, P, Q, user_map, movie_map)
    print(f"RMSE of MF = {rmse_mf:.4f}")

    mse_mf, rss_mf, tss_mf, r2_mf = evaluate_metrics_mf(test_df, P, Q, user_map, movie_map)
    print(f"MSE of MF = {mse_mf:.4f}\n"
          f"RSS of MF = {rss_mf:.4f}\n"
          f"TSS of MF = {tss_mf:.4f}\n"
          f"R2 of MF = {r2_mf:.4f}")

    hr_mf = hit_rate_mf_sample(P, Q, user_map, movie_map, train_df, test_df, N=10, max_users=100)
    print(f"MF sampling Top-10 hit rate (100 people): {hr_mf:.4f}")

    movie_titles_df = load_movie_titles()
    test_user = test_df['user_id'].iloc[0]
    show_recommendations_mf(train_df, movie_titles_df, test_user, P, Q, user_map, movie_map, top_n=10)

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
