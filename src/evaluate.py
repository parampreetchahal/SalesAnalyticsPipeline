# src/evaluate.py
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

def evaluate_and_save(best_model, X_test, y_test, model_path="models/model.joblib"):
    preds = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best_model, model_path)
    print(f"Saved best model to {model_path}")
    metrics = {"rmse": rmse, "mae": mae, "r2": r2}
    print("Evaluation metrics:", metrics)
    return metrics
