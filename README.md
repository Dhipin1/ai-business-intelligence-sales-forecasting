# 🚀 AI Business Intelligence & Sales Forecasting (Olist)

### 🔗 [**👉 VIEW LIVE DASHBOARD**](https://dhipin-bi-sales-forecasting.streamlit.app)

End-to-end Business Intelligence platform built on the Olist Brazilian E-Commerce dataset.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![DuckDB](https://img.shields.io/badge/DuckDB-FFF000?logo=duckdb&logoColor=black)
![XGBoost](https://img.shields.io/badge/XGBoost-EB5E28)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)

---

## 📊 Features
- **Automated ETL Pipeline** → DuckDB data warehouse
- **Interactive Dashboard** (Streamlit + Plotly)
- **Sales Forecasting** (XGBoost + Prophet)
- **Customer Segmentation** (RFM + KMeans)
- **KPI Reporting & Executive Overview**
- **Auto-generated AI Insights**
- **Power BI export** support

## 🛠️ Tech Stack
`Python` · `SQL` · `DuckDB` · `ETL` · `XGBoost` · `Prophet` · `scikit-learn` · `Streamlit` · `Plotly` · `Power BI`

## 🚀 Run Locally
```bash
pip install -r requirements.txt
python scripts/run_etl.py
python scripts/train_models.py
streamlit run app/streamlit_app.py