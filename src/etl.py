from __future__ import annotations

from pathlib import Path
import duckdb

REQUIRED_FILES = {
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

def _assert_files(raw_dir: Path) -> dict[str, Path]:
    paths = {}
    missing = []
    for k, fname in REQUIRED_FILES.items():
        p = raw_dir / fname
        if not p.exists():
            missing.append(fname)
        else:
            paths[k] = p
    if missing:
        raise FileNotFoundError(
            "Missing required dataset files in data/raw:\n- " + "\n- ".join(missing)
        )
    return paths

def run_etl(con: duckdb.DuckDBPyConnection, raw_dir: Path) -> None:
    paths = _assert_files(raw_dir)

    # ---------- RAW TABLES ----------
    con.execute("CREATE SCHEMA IF NOT EXISTS raw;")
    for name, p in paths.items():
        csv_path = str(p).replace("\\", "/").replace("'", "''")
        con.execute(f"""
            CREATE OR REPLACE TABLE raw.{name} AS
            SELECT * FROM read_csv_auto('{csv_path}', header=true);
        """)

    # ---------- STAGING ----------
    con.execute("CREATE SCHEMA IF NOT EXISTS stg;")
    con.execute("CREATE SCHEMA IF NOT EXISTS mart;")

    con.execute("""
        CREATE OR REPLACE TABLE stg.orders AS
        SELECT
            order_id::VARCHAR AS order_id,
            customer_id::VARCHAR AS customer_id,
            order_status::VARCHAR AS order_status,

            try_cast(order_purchase_timestamp AS TIMESTAMP) AS order_purchase_ts,
            try_cast(order_approved_at AS TIMESTAMP) AS order_approved_ts,
            try_cast(order_delivered_carrier_date AS TIMESTAMP) AS order_delivered_carrier_ts,
            try_cast(order_delivered_customer_date AS TIMESTAMP) AS order_delivered_customer_ts,
            try_cast(order_estimated_delivery_date AS TIMESTAMP) AS order_estimated_delivery_ts
        FROM raw.orders;
    """)

    con.execute("""
        CREATE OR REPLACE TABLE stg.order_items AS
        SELECT
            order_id::VARCHAR AS order_id,
            order_item_id::INTEGER AS order_item_id,
            product_id::VARCHAR AS product_id,
            seller_id::VARCHAR AS seller_id,
            try_cast(shipping_limit_date AS TIMESTAMP) AS shipping_limit_ts,
            price::DOUBLE AS price,
            freight_value::DOUBLE AS freight_value
        FROM raw.order_items;
    """)

    con.execute("""
        CREATE OR REPLACE TABLE stg.payments AS
        SELECT
            order_id::VARCHAR AS order_id,
            payment_sequential::INTEGER AS payment_sequential,
            payment_type::VARCHAR AS payment_type,
            payment_installments::INTEGER AS payment_installments,
            payment_value::DOUBLE AS payment_value
        FROM raw.payments;
    """)

    con.execute("""
        CREATE OR REPLACE TABLE stg.reviews AS
        SELECT
            review_id::VARCHAR AS review_id,
            order_id::VARCHAR AS order_id,
            review_score::INTEGER AS review_score,
            try_cast(review_creation_date AS TIMESTAMP) AS review_creation_ts,
            try_cast(review_answer_timestamp AS TIMESTAMP) AS review_answer_ts
        FROM raw.reviews;
    """)

    con.execute("""
        CREATE OR REPLACE TABLE stg.customers AS
        SELECT
            customer_id::VARCHAR AS customer_id,
            customer_unique_id::VARCHAR AS customer_unique_id,
            customer_zip_code_prefix::INTEGER AS customer_zip_prefix,
            customer_city::VARCHAR AS customer_city,
            customer_state::VARCHAR AS customer_state
        FROM raw.customers;
    """)

    con.execute("""
        CREATE OR REPLACE TABLE stg.products AS
        SELECT
            product_id::VARCHAR AS product_id,
            product_category_name::VARCHAR AS product_category_name,
            product_name_lenght::INTEGER AS product_name_length,
            product_description_lenght::INTEGER AS product_description_length,
            product_photos_qty::INTEGER AS product_photos_qty,
            product_weight_g::DOUBLE AS product_weight_g,
            product_length_cm::DOUBLE AS product_length_cm,
            product_height_cm::DOUBLE AS product_height_cm,
            product_width_cm::DOUBLE AS product_width_cm
        FROM raw.products;
    """)

    con.execute("""
        CREATE OR REPLACE TABLE stg.sellers AS
        SELECT
            seller_id::VARCHAR AS seller_id,
            seller_zip_code_prefix::INTEGER AS seller_zip_prefix,
            seller_city::VARCHAR AS seller_city,
            seller_state::VARCHAR AS seller_state
        FROM raw.sellers;
    """)

    # Geolocation is large => aggregate by zip prefix
    con.execute("""
        CREATE OR REPLACE TABLE stg.geolocation AS
        SELECT
            geolocation_zip_code_prefix::INTEGER AS zip_prefix,
            any_value(geolocation_city)::VARCHAR AS city,
            any_value(geolocation_state)::VARCHAR AS state,
            avg(geolocation_lat)::DOUBLE AS lat,
            avg(geolocation_lng)::DOUBLE AS lng
        FROM raw.geolocation
        GROUP BY 1;
    """)

    con.execute("""
        CREATE OR REPLACE TABLE stg.category_translation AS
        SELECT
            product_category_name::VARCHAR AS product_category_name,
            product_category_name_english::VARCHAR AS product_category_name_english
        FROM raw.category_translation;
    """)

    # ---------- DIMENSIONS ----------
    con.execute("""
        CREATE OR REPLACE TABLE mart.dim_customers AS
        SELECT
            c.*,
            g.lat AS customer_lat,
            g.lng AS customer_lng
        FROM stg.customers c
        LEFT JOIN stg.geolocation g
          ON c.customer_zip_prefix = g.zip_prefix;
    """)

    con.execute("""
        CREATE OR REPLACE TABLE mart.dim_sellers AS
        SELECT
            s.*,
            g.lat AS seller_lat,
            g.lng AS seller_lng
        FROM stg.sellers s
        LEFT JOIN stg.geolocation g
          ON s.seller_zip_prefix = g.zip_prefix;
    """)

    con.execute("""
        CREATE OR REPLACE TABLE mart.dim_products AS
        SELECT
            p.*,
            coalesce(t.product_category_name_english, p.product_category_name) AS product_category_en
        FROM stg.products p
        LEFT JOIN stg.category_translation t
          ON p.product_category_name = t.product_category_name;
    """)

    # ---------- FACTS / MARTS ----------
    con.execute("""
        CREATE OR REPLACE TABLE mart.order_payments AS
        SELECT
            order_id,
            sum(payment_value) AS payment_value_total,
            max(payment_installments) AS payment_installments_max,
            string_agg(DISTINCT payment_type, ', ') AS payment_types
        FROM stg.payments
        GROUP BY order_id;
    """)

    con.execute("""
        CREATE OR REPLACE TABLE mart.order_reviews AS
        SELECT
            order_id,
            avg(review_score)::DOUBLE AS review_score_avg,
            min(review_creation_ts) AS first_review_ts
        FROM stg.reviews
        GROUP BY order_id;
    """)

    con.execute("""
        CREATE OR REPLACE TABLE mart.mart_orders AS
        SELECT
            o.order_id,
            o.customer_id,
            c.customer_unique_id,
            c.customer_city,
            c.customer_state,

            o.order_status,
            o.order_purchase_ts,
            o.order_delivered_customer_ts,
            o.order_estimated_delivery_ts,

            CAST(o.order_purchase_ts AS DATE) AS order_purchase_date,

            date_diff('day', o.order_purchase_ts, o.order_delivered_customer_ts) AS delivery_days,
            date_diff('day', o.order_estimated_delivery_ts, o.order_delivered_customer_ts) AS delay_days,
            CASE
                WHEN o.order_delivered_customer_ts IS NULL THEN NULL
                WHEN o.order_estimated_delivery_ts IS NULL THEN NULL
                WHEN o.order_delivered_customer_ts > o.order_estimated_delivery_ts THEN 1
                ELSE 0
            END AS is_late,

            pay.payment_value_total,
            pay.payment_installments_max,
            pay.payment_types,

            r.review_score_avg
        FROM stg.orders o
        LEFT JOIN mart.dim_customers c ON o.customer_id = c.customer_id
        LEFT JOIN mart.order_payments pay ON o.order_id = pay.order_id
        LEFT JOIN mart.order_reviews r ON o.order_id = r.order_id;
    """)

    con.execute("""
        CREATE OR REPLACE TABLE mart.mart_order_items AS
        SELECT
            oi.order_id,
            oi.order_item_id,

            o.order_status,
            o.order_purchase_ts,
            CAST(o.order_purchase_ts AS DATE) AS order_purchase_date,

            o.customer_id,
            c.customer_unique_id,
            c.customer_city,
            c.customer_state,

            oi.product_id,
            p.product_category_en,

            oi.seller_id,
            s.seller_city,
            s.seller_state,

            oi.price,
            oi.freight_value,
            (oi.price + oi.freight_value) AS item_gmv,

            pay.payment_value_total,
            r.review_score_avg
        FROM stg.order_items oi
        LEFT JOIN stg.orders o ON oi.order_id = o.order_id
        LEFT JOIN mart.dim_customers c ON o.customer_id = c.customer_id
        LEFT JOIN mart.dim_products p ON oi.product_id = p.product_id
        LEFT JOIN mart.dim_sellers s ON oi.seller_id = s.seller_id
        LEFT JOIN mart.order_payments pay ON oi.order_id = pay.order_id
        LEFT JOIN mart.order_reviews r ON oi.order_id = r.order_id;
    """)

    con.execute("""
        CREATE OR REPLACE TABLE mart.mart_sales_daily AS
        SELECT
            order_purchase_date AS ds,
            sum(item_gmv) AS y,
            count(DISTINCT order_id) AS orders,
            count(*) AS items
        FROM mart.mart_order_items
        WHERE order_status = 'delivered'
          AND order_purchase_date IS NOT NULL
        GROUP BY 1
        ORDER BY 1;
    """)

    con.execute("""
        CREATE OR REPLACE VIEW mart.kpi_summary AS
        SELECT
            min(ds) AS min_date,
            max(ds) AS max_date,
            sum(y) AS total_gmv,
            sum(orders) AS total_orders,
            sum(items) AS total_items,
            avg(y) AS avg_daily_gmv
        FROM mart.mart_sales_daily;
    """)