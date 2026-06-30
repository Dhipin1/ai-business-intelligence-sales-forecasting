import sys
import os
from pathlib import Path

# Suppress Windows joblib physical cores warning
os.environ["LOKY_MAX_CPU_COUNT"] = os.environ.get("LOKY_MAX_CPU_COUNT", "4")

# Ensure project root is on Python path safely
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import logging

# Setup logging to track progress during training
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.config import settings
from src.db import connect
from src.forecasting import (
    train_prophet, save_prophet,
    train_xgb, save_xgb
)
from src.segmentation import compute_rfm, train_kmeans_rfm, save_segmentation


def main():
    logger.info("Starting model training pipeline...")
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    con = connect(settings.duckdb_path)

    try:
        # 1. Load Data
        logger.info("Loading daily sales data from DuckDB...")
        daily = con.execute(
            "SELECT ds, y, orders, items FROM mart.mart_sales_daily ORDER BY ds"
        ).df()

        if daily.empty:
            raise RuntimeError("mart.mart_sales_daily is empty. Run ETL first: python scripts/run_etl.py")

        # 2. Train Prophet
        logger.info("Training Prophet model...")
        prophet_model, prophet_metrics = train_prophet(daily, test_days=60)
        save_prophet(prophet_model, settings.artifacts_dir / "prophet_model.json")
        logger.info(f"Prophet MAPE: {prophet_metrics.mape:.2f}%")

        # 3. Train XGBoost
        logger.info("Training XGBoost model...")
        xgb_model, xgb_metrics, xgb_features = train_xgb(daily, test_days=60)
        save_xgb(xgb_model, xgb_features, settings.artifacts_dir / "xgb_model.joblib")
        logger.info(f"XGBoost MAPE: {xgb_metrics.mape:.2f}%")

        # 4. RFM Segmentation
        logger.info("Computing RFM and training KMeans...")
        order_items = con.execute("""
            SELECT customer_unique_id, order_id, order_purchase_date, item_gmv
            FROM mart.mart_order_items
            WHERE order_status = 'delivered'
              AND customer_unique_id IS NOT NULL
              AND order_purchase_date IS NOT NULL
        """).df()

        rfm = compute_rfm(order_items)
        scaler, km, rfm_scored = train_kmeans_rfm(rfm, n_clusters=4)
        save_segmentation(settings.artifacts_dir, scaler, km, rfm_scored)
        logger.info(f"RFM Segmentation complete. {len(rfm)} customers segmented.")

        # 5. Save Metrics
        metrics = {
            "prophet": prophet_metrics.__dict__,
            "xgboost": xgb_metrics.__dict__,
            "xgb_features": xgb_features,
            "rows_daily": int(len(daily)),
            "rows_order_items": int(len(order_items)),
            "rows_rfm": int(len(rfm)),
        }
        metrics_path = settings.artifacts_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8"
        )

        logger.info(f"✅ Training complete. Artifacts saved in: {settings.artifacts_dir}")

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise
    finally:
        con.close()


if __name__ == "__main__":
    main()