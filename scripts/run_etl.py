import sys
from pathlib import Path

# Add project root to Python path so `import src...` works
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import settings
from src.db import connect
from src.etl import run_etl

def main():
    con = connect(settings.duckdb_path)
    try:
        run_etl(con, settings.raw_data_dir)
        print(f"ETL complete. DuckDB created at: {settings.duckdb_path}")
    finally:
        con.close()

if __name__ == "__main__":
    main()