# src/model_train.py
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import numpy as np

def get_models():
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }
    return models

def cross_validate_models(X, y, models, cv_splits=5, scoring='neg_root_mean_squared_error'):
    results = {}
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=tscv, scoring=scoring, n_jobs=-1)
        results[name] = {"mean_score": np.mean(scores), "std_score": np.std(scores), "raw_scores": scores}
    return results
