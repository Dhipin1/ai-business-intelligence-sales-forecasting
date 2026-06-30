from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class Settings:
    raw_data_dir: Path = PROJECT_ROOT / os.getenv("RAW_DATA_DIR", "data/raw")
    duckdb_path: Path = PROJECT_ROOT / os.getenv("DUCKDB_PATH", "data/db/olist.duckdb")
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    exports_dir: Path = PROJECT_ROOT / "exports"

settings = Settings()