# src/feature_engineer.py
import pandas as pd
import numpy as np

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['day_of_week'] = df['date'].dt.weekday
    df['month'] = df['date'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    return df

def add_rolling_features(df: pd.DataFrame, window=7) -> pd.DataFrame:
    df = df.sort_values(['store_id', 'date'])
    df['rolling_qty_7'] = df.groupby('store_id')['total_quantity'].transform(lambda x: x.rolling(window, min_periods=1).mean())
    df['rolling_qty_30'] = df.groupby('store_id')['total_quantity'].transform(lambda x: x.rolling(30, min_periods=1).mean())
    return df

def prepare_features(df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    df = add_time_features(df)
    df = add_rolling_features(df)
    # Select features
    features = ['avg_price', 'promo_count', 'holiday_flag', 'day_of_week', 'month', 'is_weekend', 'rolling_qty_7', 'rolling_qty_30']
    X = df[features].fillna(0)
    y = df['total_quantity']
    return X, y
