# src/data_loader.py
import pandas as pd
from typing import Tuple
import sqlite3
import os

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df

def save_to_sqlite(df: pd.DataFrame, db_path: str = "data/sales.db", table_name: str = "sales"):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()

def load_from_sqlite(db_path: str = "data/sales.db", table_name: str = "sales") -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn, parse_dates=["date"])
    conn.close()
    return df

# Small convenience function that decides CSV -> SQLite -> return
def load_data(csv_path="data/synthetic_sales.csv", use_sqlite=False) -> pd.DataFrame:
    df = load_csv(csv_path)
    if use_sqlite:
        save_to_sqlite(df)
        df = load_from_sqlite()
    return df
