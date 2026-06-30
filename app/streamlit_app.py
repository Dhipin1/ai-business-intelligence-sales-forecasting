import sys
from pathlib import Path

# Ensure project root is on Python path so: `from src...` works
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import pandas as pd
import streamlit as st
import plotly.express as px

from src.config import settings
from src.db import connect
from src.forecasting import load_prophet, load_xgb
from src.features import make_time_features, make_lag_features
from src.insights import rule_based_insights

st.set_page_config(page_title="AI BI + Sales Forecasting (Olist)", layout="wide")

@st.cache_resource
def get_con():
    return connect(settings.duckdb_path)

@st.cache_data
def load_kpis():
    con = get_con()
    return con.execute("SELECT * FROM mart.kpi_summary").df()

@st.cache_data
def load_daily():
    con = get_con()
    return con.execute("SELECT ds, y, orders, items FROM mart.mart_sales_daily ORDER BY ds").df()

@st.cache_data
def load_top_categories():
    con = get_con()
    return con.execute("""
        SELECT product_category_en, sum(item_gmv) AS gmv
        FROM mart.mart_order_items
        WHERE order_status='delivered'
        GROUP BY 1
        ORDER BY gmv DESC
        LIMIT 15
    """).df()

@st.cache_data
def load_state_sales():
    con = get_con()
    return con.execute("""
        SELECT customer_state, sum(item_gmv) AS gmv, count(DISTINCT order_id) AS orders
        FROM mart.mart_order_items
        WHERE order_status='delivered'
        GROUP BY 1
        ORDER BY gmv DESC
    """).df()

def main():
    st.title("AI Business Intelligence & Sales Forecasting (Olist)")

    page = st.sidebar.radio(
        "Navigation",
        ["Overview", "Sales", "Forecast", "Customers (RFM)", "AI Insights", "Data Health"],
        index=0
    )

    kpi_df = load_kpis()
    daily = load_daily()
    daily["ds"] = pd.to_datetime(daily["ds"])

    kpi = {}
    if not kpi_df.empty:
        row = kpi_df.iloc[0].to_dict()
        kpi = {
            "min_date": str(row["min_date"]),
            "max_date": str(row["max_date"]),
            "total_gmv": float(row["total_gmv"]),
            "total_orders": int(row["total_orders"]),
            "total_items": int(row["total_items"]),
            "avg_daily_gmv": float(row["avg_daily_gmv"]),
        }

    if page == "Overview":
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total GMV (BRL)", f"{kpi.get('total_gmv', 0):,.2f}")
        c2.metric("Total Orders", f"{kpi.get('total_orders', 0):,}")
        c3.metric("Total Items", f"{kpi.get('total_items', 0):,}")
        c4.metric("Avg Daily GMV", f"{kpi.get('avg_daily_gmv', 0):,.2f}")

        st.subheader("Daily GMV (Delivered orders)")
        fig = px.line(daily, x="ds", y="y", title="GMV by Day")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Top Categories")
        top_cat = load_top_categories()
        fig2 = px.bar(top_cat, x="gmv", y="product_category_en", orientation="h")
        fig2.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig2, use_container_width=True)

    elif page == "Sales":
        st.subheader("Sales by Customer State")
        state_sales = load_state_sales()
        fig = px.bar(state_sales.head(20), x="customer_state", y="gmv", hover_data=["orders"])
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Daily Sales Table")
        st.dataframe(daily, use_container_width=True)

    elif page == "Forecast":
        st.subheader("Forecast next N days")

        horizon = st.slider("Forecast horizon (days)", 7, 180, 60)
        model_choice = st.selectbox("Model", ["Prophet", "XGBoost"], index=0)

        last_date = daily["ds"].max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
        future_df = pd.DataFrame({"ds": future_dates})

        if model_choice == "Prophet":
            model_path = settings.artifacts_dir / "prophet_model.json"
            if not model_path.exists():
                st.error("Prophet model not found. Run: python scripts/train_models.py")
                return

            m = load_prophet(model_path)
            fc = m.predict(future_df)
            out = fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()

            fig = px.line(out, x="ds", y="yhat", title="Prophet Forecast (yhat)")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(out, use_container_width=True)

        else:
            xgb_path = settings.artifacts_dir / "xgb_model.joblib"
            if not xgb_path.exists():
                st.error("XGBoost model not found. Run: python scripts/train_models.py")
                return

            pack = load_xgb(xgb_path)
            model = pack["model"]
            features = pack["features"]

            hist = daily[["ds", "y"]].copy()
            combined = pd.concat(
                [hist, pd.DataFrame({"ds": future_df["ds"], "y": [None] * horizon})],
                ignore_index=True
            ).sort_values("ds")
            combined["ds"] = pd.to_datetime(combined["ds"])

            preds = []
            for i in range(horizon):
                target_date = future_dates[i]

                temp = combined.copy()
                temp["y"] = pd.to_numeric(temp["y"], errors="coerce")

                temp2 = make_time_features(temp[["ds", "y"]].copy())
                temp2 = make_lag_features(temp2, lags=(1, 7, 14, 28))

                row = temp2[temp2["ds"] == target_date]
                if row.empty:
                    yhat = float(hist["y"].tail(7).mean())
                else:
                    yhat = float(model.predict(row[features])[0])

                preds.append(yhat)
                combined.loc[combined["ds"] == target_date, "y"] = yhat

            out = pd.DataFrame({"ds": future_dates, "yhat": preds})
            fig = px.line(out, x="ds", y="yhat", title="XGBoost Forecast (yhat)")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(out, use_container_width=True)

        metrics_path = settings.artifacts_dir / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            st.caption("Model metrics (holdout)")
            st.json(metrics)

    elif page == "Customers (RFM)":
        st.subheader("RFM Segmentation (KMeans)")

        rfm_path = settings.artifacts_dir / "rfm_scored.parquet"
        if not rfm_path.exists():
            st.error("RFM artifacts not found. Run: python scripts/train_models.py")
            return

        rfm = pd.read_parquet(rfm_path)

        seg_summary = (
            rfm.groupby("segment")[["recency", "frequency", "monetary"]]
            .mean()
            .sort_values("monetary", ascending=False)
        )
        st.dataframe(seg_summary, use_container_width=True)

        sample = rfm.sample(min(5000, len(rfm)), random_state=42)
        fig = px.scatter(
            sample, x="recency", y="monetary", color="segment",
            hover_data=["frequency"], title="RFM scatter (sample)"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("RFM Table (first 200)")
        st.dataframe(rfm.head(200), use_container_width=True)

    elif page == "AI Insights":
        st.subheader("Auto-generated Insights (rule-based)")
        ins = rule_based_insights(kpi, daily)
        for i, t in enumerate(ins, 1):
            st.write(f"{i}. {t}")

    else:
        st.subheader("Data Health Checks")
        con = get_con()
        counts = con.execute("""
            SELECT 'mart_orders' AS table, count(*) AS rows FROM mart.mart_orders
            UNION ALL
            SELECT 'mart_order_items' AS table, count(*) AS rows FROM mart.mart_order_items
            UNION ALL
            SELECT 'mart_sales_daily' AS table, count(*) AS rows FROM mart.mart_sales_daily
            UNION ALL
            SELECT 'dim_customers' AS table, count(*) AS rows FROM mart.dim_customers
            UNION ALL
            SELECT 'dim_products' AS table, count(*) AS rows FROM mart.dim_products
            UNION ALL
            SELECT 'dim_sellers' AS table, count(*) AS rows FROM mart.dim_sellers
        """).df()
        st.dataframe(counts, use_container_width=True)

        nulls = con.execute("""
            SELECT
              sum(CASE WHEN order_purchase_ts IS NULL THEN 1 ELSE 0 END) AS null_purchase_ts,
              sum(CASE WHEN customer_unique_id IS NULL THEN 1 ELSE 0 END) AS null_customer_unique_id
            FROM mart.mart_orders
        """).df()
        st.dataframe(nulls, use_container_width=True)

if __name__ == "__main__":
    main()