import sys
from pathlib import Path

# Ensure project root is on Python path so: `from src...` works
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import settings
from src.db import connect


def main():
    settings.exports_dir.mkdir(parents=True, exist_ok=True)
    con = connect(settings.duckdb_path)

    tables = {
        "dim_customers": "SELECT * FROM mart.dim_customers",
        "dim_products": "SELECT * FROM mart.dim_products",
        "dim_sellers": "SELECT * FROM mart.dim_sellers",
        "mart_orders": "SELECT * FROM mart.mart_orders",
        "mart_order_items": "SELECT * FROM mart.mart_order_items",
        "mart_sales_daily": "SELECT * FROM mart.mart_sales_daily",
    }

    try:
        for name, sql in tables.items():
            out = settings.exports_dir / f"{name}.csv"
            out_sql = str(out).replace("\\", "/")
            con.execute(f"COPY ({sql}) TO '{out_sql}' (HEADER, DELIMITER ',');")
            print("Exported:", out)

        print("Done. Load exports/*.csv into Power BI.")

    finally:
        con.close()


if __name__ == "__main__":
    main()