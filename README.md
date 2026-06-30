## AI Business Intelligence + Sales Forecasting (Olist)

### What you get
- ETL into DuckDB (local SQL database)
- BI marts: orders, items, daily sales
- Customer segmentation: RFM + KMeans
- Forecasting: Prophet + XGBoost
- Streamlit dashboard
- Optional Power BI CSV exports

### Setup
1. Put Kaggle Olist CSV files into `data/raw/`
2. Create venv and install:
   - `pip install -r requirements.txt`

### Run ETL
`python scripts/run_etl.py`

### Train models
`python scripts/train_models.py`

### Run dashboard
`streamlit run app/streamlit_app.py`

### Power BI
Run `python scripts/export_powerbi.py` and load CSVs from `exports/` into Power BI.