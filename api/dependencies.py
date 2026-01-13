"""Dependency injection for models and shared resources."""

import json
import os
from typing import Optional

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from api.config import get_settings


class ModelManager:
    """Manages XGBoost model loading and inference."""

    def __init__(self):
        self.model_point: Optional[XGBRegressor] = None
        self.model_lower: Optional[XGBRegressor] = None
        self.model_upper: Optional[XGBRegressor] = None
        self.feature_cols: list[str] = []
        self.model_config: dict = {}
        self._loaded = False

    def load_models(self) -> None:
        """Load all models and configuration from artifacts directory."""
        settings = get_settings()
        artifacts_dir = settings.artifacts_dir

        # Load point prediction model
        point_path = os.path.join(artifacts_dir, "model_point.json")
        if os.path.exists(point_path):
            self.model_point = XGBRegressor()
            self.model_point.load_model(point_path)
            print(f"Loaded point prediction model from {point_path}")
        else:
            raise FileNotFoundError(f"Point model not found at {point_path}")

        # Load lower bound model
        lower_path = os.path.join(artifacts_dir, "model_lower.json")
        if os.path.exists(lower_path):
            self.model_lower = XGBRegressor()
            self.model_lower.load_model(lower_path)
            print(f"Loaded lower bound model from {lower_path}")
        else:
            raise FileNotFoundError(f"Lower bound model not found at {lower_path}")

        # Load upper bound model
        upper_path = os.path.join(artifacts_dir, "model_upper.json")
        if os.path.exists(upper_path):
            self.model_upper = XGBRegressor()
            self.model_upper.load_model(upper_path)
            print(f"Loaded upper bound model from {upper_path}")
        else:
            raise FileNotFoundError(f"Upper bound model not found at {upper_path}")

        # Load feature columns
        feature_cols_path = os.path.join(artifacts_dir, "feature_cols.json")
        if os.path.exists(feature_cols_path):
            with open(feature_cols_path, "r") as f:
                self.feature_cols = json.load(f)
            print(f"Loaded {len(self.feature_cols)} feature columns")
        else:
            raise FileNotFoundError(f"Feature columns not found at {feature_cols_path}")

        # Load model config
        config_path = os.path.join(artifacts_dir, "model_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.model_config = json.load(f)
            print(f"Loaded model config: {self.model_config}")

        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        """Check if all models are loaded."""
        return self._loaded

    def predict(self, features: np.ndarray) -> tuple[float, float, float]:
        """
        Generate predictions from all three models.

        Args:
            features: Feature vector of shape (1, n_features)

        Returns:
            Tuple of (point_prediction, lower_bound, upper_bound) in price space (USD)
        """
        if not self._loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        # Predictions are in log space
        log_point = self.model_point.predict(features)[0]
        log_lower = self.model_lower.predict(features)[0]
        log_upper = self.model_upper.predict(features)[0]

        # Transform to price space
        price_point = float(np.exp(log_point))
        price_lower = float(np.exp(log_lower))
        price_upper = float(np.exp(log_upper))

        return price_point, price_lower, price_upper


class NeighborLookup:
    """Manages neighbor lookup for similarity features."""

    def __init__(self):
        self.neighbors_df: Optional[pd.DataFrame] = None
        self._loaded = False

    def load(self, filepath: str) -> None:
        """Load neighbors from parquet file."""
        if os.path.exists(filepath):
            self.neighbors_df = pd.read_parquet(filepath)
            self.neighbors_df = self.neighbors_df.rename(
                columns={"neighbors": "neighbor_id"}
            )
            print(f"Loaded {len(self.neighbors_df)} neighbor mappings")
            self._loaded = True
        else:
            print(f"Warning: Neighbors file not found at {filepath}")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def get_neighbors(self, gemrate_id: str, n_neighbors: int = 5) -> list[dict]:
        """
        Get top N neighbors for a card.

        Returns list of dicts with neighbor_id and score.
        """
        if not self._loaded or self.neighbors_df is None:
            return []

        matches = self.neighbors_df[self.neighbors_df["gemrate_id"] == gemrate_id]
        matches = matches.head(n_neighbors)

        return [
            {"neighbor_id": row["neighbor_id"], "score": row["score"]}
            for _, row in matches.iterrows()
        ]


# Global instances (singleton pattern)
model_manager = ModelManager()
neighbor_lookup = NeighborLookup()


def get_model_manager() -> ModelManager:
    """Get the model manager instance."""
    return model_manager


def get_neighbor_lookup() -> NeighborLookup:
    """Get the neighbor lookup instance."""
    return neighbor_lookup
