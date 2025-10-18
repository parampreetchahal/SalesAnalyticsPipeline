# generate_data.py
"""
Generate a synthetic sales dataset for demonstration purposes.
Creates data/synthetic_sales.csv
"""
import os
import numpy as np
import pandas as pd

def generate_sales_data(out_path="data/synthetic_sales.csv",
                        n_stores=10,
                        n_days=365*2,  # two years
                        seed=42):
    np.random.seed(seed)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    rows = []
    date_range = pd.date_range(end=pd.Timestamp.today(), periods=n_days).sort_values()
    product_ids = [f"P{str(i).zfill(3)}" for i in range(1, 21)]  # 20 products

    for store in range(1, n_stores + 1):
        store_id = f"S{store:03d}"
        base_sales = np.random.uniform(50, 200)  # base daily qty
        seasonality = 20 * np.sin(np.linspace(0, 3 * np.pi, n_days))  # seasonal component
        for i, date in enumerate(date_range):
            for prod in product_ids:
                price = np.round(np.random.uniform(5, 60), 2)
                promo = np.random.binomial(1, 0.05)  # 5% chance of promo
                holiday = int(date.weekday() >= 5)  # weekend flag (0-1)
                noise = np.random.normal(0, 10)
                qty = max(0, base_sales + seasonality[i] * np.random.uniform(0.5, 1.5) + noise - price*0.1 + promo*10 - holiday*5)
                rows.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "store_id": store_id,
                    "product_id": prod,
                    "price": price,
                    "promo": promo,
                    "holiday": holiday,
                    "quantity": int(np.round(qty)),
                })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Generated synthetic dataset at: {out_path}")
    return df

if __name__ == "__main__":
    generate_sales_data()
