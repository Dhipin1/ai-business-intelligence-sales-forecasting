import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.config import settings

# Folder for sample data (small, goes to GitHub)
SAMPLE_DIR = ROOT / "data" / "sample"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

RAW = settings.raw_data_dir

def sample_orders():
    orders = pd.read_csv(RAW / "olist_orders_dataset.csv")
    # Keep a manageable number of orders
    orders = orders.sample(min(15000, len(orders)), random_state=42)
    return orders

def main():
    orders = sample_orders()
    keep_ids = set(orders["order_id"])
    orders.to_csv(SAMPLE_DIR / "olist_orders_dataset.csv", index=False)

    # Filter related tables by order_id
    for fname, key in [
        ("olist_order_items_dataset.csv", "order_id"),
        ("olist_order_payments_dataset.csv", "order_id"),
        ("olist_order_reviews_dataset.csv", "order_id"),
    ]:
        df = pd.read_csv(RAW / fname)
        df = df[df[key].isin(keep_ids)]
        df.to_csv(SAMPLE_DIR / fname, index=False)

    # Customers needed for these orders
    cust_ids = set(orders["customer_id"])
    customers = pd.read_csv(RAW / "olist_customers_dataset.csv")
    customers = customers[customers["customer_id"].isin(cust_ids)]
    customers.to_csv(SAMPLE_DIR / "olist_customers_dataset.csv", index=False)

    # Products + sellers from order_items
    items = pd.read_csv(SAMPLE_DIR / "olist_order_items_dataset.csv")
    prod_ids = set(items["product_id"])
    seller_ids = set(items["seller_id"])

    products = pd.read_csv(RAW / "olist_products_dataset.csv")
    products = products[products["product_id"].isin(prod_ids)]
    products.to_csv(SAMPLE_DIR / "olist_products_dataset.csv", index=False)

    sellers = pd.read_csv(RAW / "olist_sellers_dataset.csv")
    sellers = sellers[sellers["seller_id"].isin(seller_ids)]
    sellers.to_csv(SAMPLE_DIR / "olist_sellers_dataset.csv", index=False)

    # Small files - copy fully
    pd.read_csv(RAW / "product_category_name_translation.csv").to_csv(
        SAMPLE_DIR / "product_category_name_translation.csv", index=False
    )

    # Geolocation - sample to keep small
    geo = pd.read_csv(RAW / "olist_geolocation_dataset.csv")
    geo = geo.sample(min(50000, len(geo)), random_state=42)
    geo.to_csv(SAMPLE_DIR / "olist_geolocation_dataset.csv", index=False)

    print("Sample created in:", SAMPLE_DIR)
    for f in SAMPLE_DIR.glob("*.csv"):
        print(f.name, round(f.stat().st_size / 1_000_000, 2), "MB")

if __name__ == "__main__":
    main()