# src/preprocess.py
import pandas as pd
from typing import Tuple

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Ensure date is datetime
    if df['date'].dtype == 'O':
        df['date'] = pd.to_datetime(df['date'])
    # Remove impossible values
    df = df[df['quantity'] >= 0]
    df = df[df['price'] >= 0]
    # Fill missing columns if any
    for col in ['promo', 'holiday']:
        if col not in df.columns:
            df[col] = 0
    # Basic dtypes
    df['promo'] = df['promo'].astype(int)
    df['holiday'] = df['holiday'].astype(int)
    return df

def aggregate_daily_store(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate data to daily store-level sales (sum across products).
    """
    df_agg = df.groupby(['date', 'store_id']).agg(
        total_quantity=('quantity', 'sum'),
        avg_price=('price', 'mean'),
        promo_count=('promo', 'sum'),
        holiday_flag=('holiday', 'max')
    ).reset_index()
    return df_agg
