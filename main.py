
"""
Orchestrator script for the Sales Analytics & Forecasting Pipeline.

Steps:
1. Generate dataset (optional: run generate_data.py separately)
2. Load CSV
3. Preprocess & aggregate
4. EDA
5. Feature engineering
6. Train baseline models and evaluate
7. Save best model
"""

import warnings
import numpy as np

warnings.filterwarnings("ignore")

from src.data_loader import load_data
from src.preprocess import basic_clean, aggregate_daily_store
from src.eda import summary_stats, plot_time_series, plot_correlation
from src.feature_engineer import prepare_features
from src.model_train import get_models, cross_validate_models
from src.evaluate import evaluate_and_save

from sklearn.model_selection import train_test_split

def run_pipeline(csv_path="data/synthetic_sales.csv", use_sqlite=False):
    # 1) Load
    df = load_data(csv_path, use_sqlite=use_sqlite)
    print("Loaded raw rows:", len(df))

    # 2) Clean
    df_clean = basic_clean(df)
    print("After basic clean rows:", len(df_clean))

    # 3) Aggregate to daily-store
    df_agg = aggregate_daily_store(df_clean)
    print("After aggregation rows:", len(df_agg))

    # 4) EDA
    summary_stats(df_agg)
    plot_time_series(df_agg)
    plot_correlation(df_agg)

    # 5) Feature engineering
    X, y = prepare_features(df_agg)
    print("Feature matrix shape:", X.shape)

    # 6) Train / CV
    models = get_models()
    cv_results = cross_validate_models(X, y, models, cv_splits=5)
    print("CV Results (neg RMSE):")
    for k, v in cv_results.items():
        print(f"{k}: mean={v['mean_score']:.4f}, std={v['std_score']:.4f}")

    # 7) Train-test split for final eval
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    best_name, best_model, best_rmse = None, None, float('inf')
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = np.sqrt(((preds - y_test) ** 2).mean())
        print(f"{name} RMSE on test: {rmse:.4f}")
        if rmse < best_rmse:
            best_name, best_model, best_rmse = name, model, rmse

    print(f"Best model: {best_name} with RMSE={best_rmse:.4f}")

    # 8) Save best model & evaluate metrics
    metrics = evaluate_and_save(best_model, X_test, y_test)
    return metrics

if __name__ == "__main__":
    run_pipeline()
