from __future__ import annotations
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

def compute_rfm(order_items: pd.DataFrame, as_of_date=None) -> pd.DataFrame:
    """
    order_items columns:
      customer_unique_id, order_id, order_purchase_date, item_gmv
    """
    df = order_items.copy()
    df["order_purchase_date"] = pd.to_datetime(df["order_purchase_date"])

    if as_of_date is None:
        as_of_date = df["order_purchase_date"].max() + pd.Timedelta(days=1)

    g = df.groupby("customer_unique_id").agg(
        last_purchase=("order_purchase_date", "max"),
        frequency=("order_id", "nunique"),
        monetary=("item_gmv", "sum"),
    ).reset_index()

    g["recency"] = (pd.to_datetime(as_of_date) - g["last_purchase"]).dt.days
    rfm = g[["customer_unique_id", "recency", "frequency", "monetary"]].copy()
    return rfm

def train_kmeans_rfm(rfm: pd.DataFrame, n_clusters=4, random_state=42):
    X = rfm[["recency", "frequency", "monetary"]].copy()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
    labels = km.fit_predict(Xs)

    out = rfm.copy()
    out["segment"] = labels
    return scaler, km, out

def save_segmentation(artifacts_dir, scaler, model, rfm_scored: pd.DataFrame):
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, artifacts_dir / "rfm_scaler.joblib")
    joblib.dump(model, artifacts_dir / "rfm_kmeans.joblib")
    rfm_scored.to_parquet(artifacts_dir / "rfm_scored.parquet", index=False)