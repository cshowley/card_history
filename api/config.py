"""Configuration settings for the Card Price Prediction API."""

import os
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # MongoDB
    mongo_uri: str = os.getenv("MONGO_URI", "")
    mongo_db_name: str = "gemrate"
    ebay_collection: str = "ebay_graded_items"
    pwcc_collection: str = "pwcc_graded_items"

    # Model artifacts
    artifacts_dir: str = "api/artifacts"
    neighbors_file: str = "api/artifacts/neighbors.parquet"

    # Index API
    index_api_url: str = "https://price.collectorcrypt.com/api/indexes/modern"
    index_cache_ttl_seconds: int = 3600  # 1 hour

    # Feature configuration
    n_sales_back: int = 5
    n_neighbors: int = 5

    # API settings
    api_title: str = "Card Price Prediction API"
    api_version: str = "1.0.0"
    api_description: str = "Predict Pokemon card prices with 95% confidence intervals"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra env vars like MONGO_URI


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
