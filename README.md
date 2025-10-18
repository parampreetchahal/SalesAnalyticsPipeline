# Sales Analytics Pipeline

## Overview
This project is a **general sales analytics pipeline** that allows you to analyze sales data, generate visual reports, and build predictive models for forecasting sales. It is designed to handle CSV datasets and includes preprocessing, aggregation, visualization, and machine learning components.

---

## Features
- Load and clean raw sales data
- Aggregate sales data by store and date
- Generate visual reports:
  - Time-series plots
  - Correlation heatmaps
- Feature engineering for predictive modeling
- Train and evaluate models:
  - Linear Regression
  - Ridge Regression
  - Random Forest Regression
- Evaluate model performance using RMSE

---

## Dataset
- The project expects a CSV file containing sales data.
- Example columns:
  - `date` — date of transaction
  - `store_id` — unique store identifier
  - `product_id` — unique product identifier
  - `quantity` — units sold
  - `sales` — revenue
  - `promo_count` — number of promotions
  - `holiday_flag` — 1 if holiday, 0 otherwise

> You can use synthetic or real-world sales datasets. Example file: `data/synthetic_sales.csv`

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/username/sales-analytics-pipeline.git
cd sales-analytics-pipeline
```
## Project Structure
```bash
sales-analytics-pipeline/
│
├── data/                 # CSV datasets
├── reports/              # Generated plots and reports
├── src/                  # Source code modules
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── modeling.py
│   └── utils.py
├── main.py               # Entry point for pipeline
├── requirements.txt      # Python dependencies
└── README.md
```

## License
This project is licensed under the MIT License.

## Author
Parampreet Singh
GitHub: github.com/parampreetchahal
LinkedIn: linkedin.com/in/parampreet-singh23
Email: parampreets537@gmail.com
git clone https://github.com/username/sales-analytics-pipeline.git
cd sales-analytics-pipeline
