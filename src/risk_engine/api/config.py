from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RISK_", env_file=".env")
    returns_path: Path = Path("data/processed/returns_matrix.parquet")
    db_path: Path = Path("data/processed/risk_engine.db")
    api_key: str = "dev-key-change-me"
    cors_origins: list[str] = ["http://localhost:8501"]  # Streamlit's default port
