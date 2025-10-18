# src/eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    desc = df.describe(include='all')
    print("---- Summary Stats ----")
    print(desc)
    return desc

def plot_time_series(df: pd.DataFrame, out_dir="reports"):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(10, 4))
    df.groupby('date')['total_quantity'].sum().plot(title="Total Quantity Sold Over Time")
    plt.ylabel("Total Quantity")
    plt.tight_layout()
    path = os.path.join(out_dir, "time_series_total_quantity.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved time-series plot to {path}")

def plot_correlation(df: pd.DataFrame, out_dir="reports"):
    os.makedirs(out_dir, exist_ok=True)
    numeric = df.select_dtypes(include=['int64', 'float64'])
    corr = numeric.corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag")
    plt.title("Feature Correlation")
    plt.tight_layout()
    path = os.path.join(out_dir, "correlation_heatmap.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved correlation heatmap to {path}")
