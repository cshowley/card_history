"""Feature builder for constructing model input from API request."""

from typing import Optional

import numpy as np

from api.config import get_settings
from api.dependencies import get_neighbor_lookup
from api.models import GradingCompany, PredictionRequest, Source
from api.services.index_cache import get_index_cache
from api.services.mongodb_client import get_mongodb_client


class FeatureBuilder:
    """Builds feature vectors for model inference."""

    def __init__(self, feature_cols: list[str]):
        self.feature_cols = feature_cols
        self.n_features = len(feature_cols)

    def build_features(
        self, request: PredictionRequest
    ) -> tuple[np.ndarray, list[str]]:
        """
        Build feature vector from prediction request.

        Returns:
            Tuple of (feature_array, warnings)
        """
        warnings = []
        features = {col: np.nan for col in self.feature_cols}

        # Basic features from request
        features["grade"] = float(request.grade)
        features["half_grade"] = 1.0 if request.half_grade else 0.0

        # Source encoding
        features["source_ebay"] = 1 if request.source == Source.EBAY else 0
        features["source_fanatics_weekly"] = (
            1 if request.source == Source.FANATICS_WEEKLY else 0
        )
        features["source_fanatics_premier"] = (
            1 if request.source == Source.FANATICS_PREMIER else 0
        )

        # Grading company encoding
        for gc in ["BCCG", "BGS", "BVG", "CGC", "CSG", "PSA", "SGC", "Unknown"]:
            col = f"grade_co_{gc}"
            if col in features:
                features[col] = 0

        gc_map = {
            GradingCompany.PSA: "PSA",
            GradingCompany.CGC: "CGC",
            GradingCompany.BGS: "BGS",
        }
        gc_col = f"grade_co_{gc_map.get(request.grading_company, 'Unknown')}"
        if gc_col in features:
            features[gc_col] = 1

        # Number of bids (default to 0 for predictions)
        features["number_of_bids"] = 0

        # Seller popularity (use average as default)
        features["seller_popularity"] = 0.5

        # Get historical sales data from MongoDB
        mongo = get_mongodb_client()
        settings = get_settings()

        # Previous sales for same card/grade
        prev_sales = mongo.get_previous_sales(
            request.gemrate_id, float(request.grade), settings.n_sales_back
        )

        if not prev_sales:
            warnings.append("No previous sales found for this card/grade")

        for i, sale in enumerate(prev_sales, 1):
            prefix = f"prev_{i}"
            features[f"{prefix}_price"] = sale.get("price")
            features[f"{prefix}_half_grade"] = sale.get("half_grade")
            features[f"{prefix}_grade_co_BGS"] = sale.get("grade_co_BGS")
            features[f"{prefix}_grade_co_CGC"] = sale.get("grade_co_CGC")
            features[f"{prefix}_grade_co_PSA"] = sale.get("grade_co_PSA")
            features[f"{prefix}_days_ago"] = sale.get("days_ago")

        # Adjacent grade sales (above)
        above_sales = mongo.get_adjacent_grade_sales(
            request.gemrate_id, float(request.grade), "above", settings.n_sales_back
        )
        for i, sale in enumerate(above_sales, 1):
            prefix = f"prev_{i}_above"
            features[f"{prefix}_price"] = sale.get("price")
            features[f"{prefix}_half_grade"] = sale.get("half_grade")
            features[f"{prefix}_grade_co_BGS"] = sale.get("grade_co_BGS")
            features[f"{prefix}_grade_co_CGC"] = sale.get("grade_co_CGC")
            features[f"{prefix}_grade_co_PSA"] = sale.get("grade_co_PSA")
            features[f"{prefix}_seller_popularity"] = 0.5  # Default
            features[f"{prefix}_days_ago"] = sale.get("days_ago")

        # Adjacent grade sales (below)
        below_sales = mongo.get_adjacent_grade_sales(
            request.gemrate_id, float(request.grade), "below", settings.n_sales_back
        )
        for i, sale in enumerate(below_sales, 1):
            prefix = f"prev_{i}_below"
            features[f"{prefix}_price"] = sale.get("price")
            features[f"{prefix}_half_grade"] = sale.get("half_grade")
            features[f"{prefix}_grade_co_BGS"] = sale.get("grade_co_BGS")
            features[f"{prefix}_grade_co_CGC"] = sale.get("grade_co_CGC")
            features[f"{prefix}_grade_co_PSA"] = sale.get("grade_co_PSA")
            features[f"{prefix}_seller_popularity"] = 0.5  # Default
            features[f"{prefix}_days_ago"] = sale.get("days_ago")

        # Weekly average features (use recent sales average as proxy)
        if prev_sales:
            avg_price = np.mean([s["price"] for s in prev_sales])
            for weeks in [1, 2, 3, 4]:
                features[f"avg_price_{weeks}w_ago"] = avg_price
                features[f"avg_half_grade_{weeks}w_ago"] = features["half_grade"]
                features[f"avg_seller_popularity_{weeks}w_ago"] = 0.5
                features[f"avg_grade_co_BGS_{weeks}w_ago"] = features.get(
                    "grade_co_BGS", 0
                )
                features[f"avg_grade_co_CGC_{weeks}w_ago"] = features.get(
                    "grade_co_CGC", 0
                )
                features[f"avg_grade_co_PSA_{weeks}w_ago"] = features.get(
                    "grade_co_PSA", 0
                )

        # Neighbor features
        neighbor_lookup = get_neighbor_lookup()
        neighbors = neighbor_lookup.get_neighbors(
            request.gemrate_id, settings.n_neighbors
        )

        if neighbors:
            neighbor_ids = [n["neighbor_id"] for n in neighbors]
            neighbor_prices = mongo.get_neighbor_avg_prices(
                neighbor_ids, float(request.grade)
            )

            for i, neighbor in enumerate(neighbors, 1):
                nid = neighbor["neighbor_id"]
                features[f"neighbor_{i}_similarity"] = neighbor["score"]
                features[f"neighbor_{i}_avg_price"] = neighbor_prices.get(nid, np.nan)
        else:
            warnings.append("No neighbor cards found")

        # Index features
        index_cache = get_index_cache()
        index_features = index_cache.get_index_features()
        for key, value in index_features.items():
            if key in features:
                features[key] = value

        # Build feature array in correct order
        feature_array = np.array(
            [[features.get(col, np.nan) for col in self.feature_cols]]
        )

        # Replace NaN with 0 for model inference
        feature_array = np.nan_to_num(feature_array, nan=0.0)

        return feature_array, warnings


# Global instance
_feature_builder: Optional[FeatureBuilder] = None


def get_feature_builder() -> FeatureBuilder:
    """Get singleton feature builder instance."""
    global _feature_builder
    if _feature_builder is None:
        from api.dependencies import get_model_manager

        model_manager = get_model_manager()
        _feature_builder = FeatureBuilder(model_manager.feature_cols)
    return _feature_builder


def initialize_feature_builder(feature_cols: list[str]) -> FeatureBuilder:
    """Initialize feature builder with feature columns."""
    global _feature_builder
    _feature_builder = FeatureBuilder(feature_cols)
    return _feature_builder
