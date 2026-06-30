import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
import plotly.express as px

from src.config import settings
from src.db import connect
from src.features import make_time_features, make_lag_features
from src.insights import rule_based_insights

st.set_page_config(page_title="AI BI + Sales Forecasting (Olist)", layout="wide")


@st.cache_resource
def get_con():
    """
    Open DB connection. If the 'mart' schema doesn't exist,
    build everything from sample data on this SAME connection.
    """
    con = connect(settings.duckdb_path)

    # Check if mart schema exists
    try:
        con.execute("SELECT 1 FROM mart.kpi_summary LIMIT 1").fetchall()
        return con  # already built
    except Exception:
        pass  # need to build

    # ---- Build ETL on this connection ----
    from src.etl import run_etl
    sample_dir = ROOT / "data" / "sample"
    raw_dir = sample_dir if sample_dir.exists() else settings.raw_data_dir
    run_etl(con, raw_dir)

    # ---- Train + save models ----
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)

    try:
        from src.forecasting import train_xgb, save_xgb
        daily = con.execute(
            "SELECT ds, y, orders, items FROM mart.mart_sales_daily ORDER BY ds"
        ).df()
        if not daily.empty:
            xgb_model, _, feats = train_xgb(daily, test_days=30)
            save_xgb(xgb_model, feats, settings.artifacts_dir / "xgb_model.joblib")
    except Exception as e:
        print("XGB training skipped:", e)

    try:
        from src.segmentation import compute_rfm, train_kmeans_rfm, save_segmentation
        order_items = con.execute("""
            SELECT customer_unique_id, order_id, order_purchase_date, item_gmv
            FROM mart.mart_order_items
            WHERE order_status='delivered'
              AND customer_unique_id IS NOT NULL
              AND order_purchase_date IS NOT NULL
        """).df()
        if not order_items.empty:
            rfm = compute_rfm(order_items)
            scaler, km, rfm_scored = train_kmeans_rfm(rfm, n_clusters=4)
            save_segmentation(settings.artifacts_dir, scaler, km, rfm_scored)
    except Exception as e:
        print("RFM training skipped:", e)

    return con


@st.cache_data
def load_kpis():
    return get_con().execute("SELECT * FROM mart.kpi_summary").df()

@st.cache_data
def load_daily():
    return get_con().execute(
        "SELECT ds, y, orders, items FROM mart.mart_sales_daily ORDER BY ds"
    ).df()

@st.cache_data
def load_top_categories():
    return get_con().execute("""
        SELECT product_category_en, sum(item_gmv) AS gmv
        FROM mart.mart_order_items
        WHERE order_status='delivered'
        GROUP BY 1 ORDER BY gmv DESC LIMIT 15
    """).df()

@st.cache_data
def load_state_sales():
    return get_con().execute("""
        SELECT customer_state, sum(item_gmv) AS gmv, count(DISTINCT order_id) AS orders
        FROM mart.mart_order_items
        WHERE order_status='delivered'
        GROUP BY 1 ORDER BY gmv DESC
    """).df()


def load_xgb_local():
    import joblib
    return joblib.load(settings.artifacts_dir / "xgb_model.joblib")


def main():
    # Trigger DB build before anything else
    get_con()

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
        st.plotly_chart(px.line(daily, x="ds", y="y", title="GMV by Day"),
                        use_container_width=True)

        st.subheader("Top Categories")
        top_cat = load_top_categories()
        fig2 = px.bar(top_cat, x="gmv", y="product_category_en", orientation="h")
        fig2.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig2, use_container_width=True)

    elif page == "Sales":
        st.subheader("Sales by Customer State")
        state_sales = load_state_sales()
        st.plotly_chart(
            px.bar(state_sales.head(20), x="customer_state", y="gmv", hover_data=["orders"]),
            use_container_width=True
        )
        st.subheader("Daily Sales Table")
        st.dataframe(daily, use_container_width=True)

    elif page == "Forecast":
        st.subheader("Forecast next N days (XGBoost)")
        horizon = st.slider("Forecast horizon (days)", 7, 180, 60)

        xgb_path = settings.artifacts_dir / "xgb_model.joblib"
        if not xgb_path.exists():
            st.error("XGBoost model not found.")
            return

        pack = load_xgb_local()
        model = pack["model"]
        features = pack["features"]

        last_date = daily["ds"].max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

        hist = daily[["ds", "y"]].copy()
        combined = pd.concat(
            [hist, pd.DataFrame({"ds": future_dates, "y": [None] * horizon})],
            ignore_index=True
        ).sort_values("ds")
        combined["ds"] = pd.to_datetime(combined["ds"])

        preds = []
        for i in range(horizon):
            target_date = future_dates[i]
            temp = combined.copy()
            temp["y"] = pd.to_numeric(temp["y"], errors="coerce")
            t2 = make_time_features(temp[["ds", "y"]].copy())
            t2 = make_lag_features(t2, lags=(1, 7, 14, 28))
            row = t2[t2["ds"] == target_date]
            yhat = float(hist["y"].tail(7).mean()) if row.empty else float(model.predict(row[features])[0])
            preds.append(yhat)
            combined.loc[combined["ds"] == target_date, "y"] = yhat

        out = pd.DataFrame({"ds": future_dates, "yhat": preds})
        st.plotly_chart(px.line(out, x="ds", y="yhat", title="XGBoost Forecast"),
                        use_container_width=True)
        st.dataframe(out, use_container_width=True)

    elif page == "Customers (RFM)":
        st.subheader("RFM Segmentation (KMeans)")
        rfm_path = settings.artifacts_dir / "rfm_scored.parquet"
        if not rfm_path.exists():
            st.error("RFM artifacts not found.")
            return
        rfm = pd.read_parquet(rfm_path)
        seg = (rfm.groupby("segment")[["recency", "frequency", "monetary"]]
               .mean().sort_values("monetary", ascending=False))
        st.dataframe(seg, use_container_width=True)
        sample = rfm.sample(min(5000, len(rfm)), random_state=42)
        st.plotly_chart(
            px.scatter(sample, x="recency", y="monetary", color="segment",
                       hover_data=["frequency"], title="RFM scatter (sample)"),
            use_container_width=True
        )

    elif page == "AI Insights":
        st.subheader("Auto-generated Insights")
        for i, t in enumerate(rule_based_insights(kpi, daily), 1):
            st.write(f"{i}. {t}")

    else:
        st.subheader("Data Health Checks")
        counts = get_con().execute("""
            SELECT 'mart_orders' AS t, count(*) AS rows FROM mart.mart_orders
            UNION ALL SELECT 'mart_order_items', count(*) FROM mart.mart_order_items
            UNION ALL SELECT 'mart_sales_daily', count(*) FROM mart.mart_sales_daily
            UNION ALL SELECT 'dim_customers', count(*) FROM mart.dim_customers
        """).df()
        st.dataframe(counts, use_container_width=True)


if __name__ == "__main__":
    main()