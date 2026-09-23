from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RISK_", env_file=".env")
    returns_path: Path = Path("data/processed/returns_matrix.parquet")
    db_path: Path = Path("data/processed/risk_engine.db")
