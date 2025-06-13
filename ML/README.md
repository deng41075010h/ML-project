# Recommender Systems with Netflix & MovieLens Datasets

This project implements several recommendation-system algorithms on two datasets:  
the **Netflix Prize Dataset** and the **MovieLens Dataset**, along with evaluation metrics and result.

---

## How to Run

### 1. Netflix Dataset

- **`main.py`**  
  Implements **CBF**, **User-based CF**, **Item-based CF**, and **Matrix Factorization (MF)**.  
  Also includes evaluation metrics (RMSE, MSE, RSS, TSS, R²), **Hit Rate @ 10**, and recommendation output.

  ```bash
  # Run
  python main.py
  ```

- **`main_cv.py`**  
  Performs **5-Fold Cross-Validation** on CF methods.  
  Reports metrics per fold and averages across all folds.

  ```bash
  # Run
  python main_cv.py
  ```

### 2. MovieLens Dataset

- **`cbf_main.py`** — Content-Based Filtering **without** considering time.  

  ```bash
  python main.py
  ```

- **`cbf_main.py`** — Time-aware Content-Based Filtering.  

  ```bash
  python main.py
  ```

---

## Evaluation Metrics

Implemented metrics:

- **RMSE**, **MSE**  
- **RSS**, **TSS**  
- **R² Score**  
- **Hit Rate @ Top-N**  
- **Cross-Validation metrics (5-Fold)**

---

## Dataset Sources

- **Netflix Prize Dataset**  
  <https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data>

- **MovieLens Dataset**  
  <https://grouplens.org/datasets/movielens/>