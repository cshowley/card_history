"""MongoDB client for historical sales data queries."""

import re
from datetime import datetime
from typing import Optional

import numpy as np
from pymongo import MongoClient
from pymongo.database import Database

from api.config import get_settings


class MongoDBClient:
    """Client for querying historical sales from MongoDB."""

    def __init__(self):
        self.client: Optional[MongoClient] = None
        self.db: Optional[Database] = None
        self._connected = False

    def connect(self) -> None:
        """Establish MongoDB connection."""
        settings = get_settings()
        if not settings.mongo_uri:
            raise ValueError("MONGO_URI environment variable not set")

        self.client = MongoClient(settings.mongo_uri)
        self.db = self.client[settings.mongo_db_name]
        self._connected = True
        print(f"Connected to MongoDB database: {settings.mongo_db_name}")

    def disconnect(self) -> None:
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _clean_grade(self, val) -> Optional[float]:
        """Clean grade string to numeric value."""
        s = str(val).lower().strip().replace("g", "").replace("_", ".")
        if s in ["nan", "none", "", "0", "auth"]:
            return None

        if "10b" in s or "10black" in s:
            return 11.0

        if any(x in s for x in ["pristine", "perfect", "10p"]):
            return 10.5

        match = re.search(r"(\d+(\.\d+)?)", s)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None

    def _parse_price(self, price_str: str) -> Optional[float]:
        """Parse price string to float."""
        try:
            cleaned = re.sub(r"[^\d.]", "", str(price_str))
            return float(cleaned) if cleaned else None
        except (ValueError, TypeError):
            return None

    def get_previous_sales(
        self,
        gemrate_id: str,
        grade: float,
        n_sales: int = 5,
        before_date: Optional[datetime] = None,
    ) -> list[dict]:
        """
        Get previous sales for a card at a specific grade.

        Returns list of dicts with: price (log), date, days_ago, grading_company flags
        """
        if not self._connected:
            return []

        settings = get_settings()
        before_date = before_date or datetime.now()

        # Query eBay sales
        ebay_collection = self.db[settings.ebay_collection]
        ebay_pipeline = [
            {
                "$match": {
                    "$or": [
                        {"gemrate_data.universal_gemrate_id": gemrate_id},
                        {"gemrate_data.gemrate_id": gemrate_id},
                    ],
                    "item_data.date": {"$lt": before_date},
                }
            },
            {"$sort": {"item_data.date": -1}},
            {"$limit": n_sales * 3},  # Get extra to filter by grade
            {
                "$project": {
                    "date": "$item_data.date",
                    "price": "$item_data.price",
                    "grade": "$gemrate_data.grade",
                    "grading_company": 1,
                }
            },
        ]

        results = []
        try:
            for doc in ebay_collection.aggregate(ebay_pipeline):
                parsed_grade = self._clean_grade(doc.get("grade"))
                if parsed_grade is None:
                    continue

                grade_floor = np.floor(parsed_grade)
                if grade_floor != grade:
                    continue

                price = self._parse_price(doc.get("price"))
                if price is None or price <= 0:
                    continue

                sale_date = doc.get("date")
                if sale_date is None:
                    continue

                days_ago = (before_date - sale_date).days if sale_date else None

                grading_co = str(doc.get("grading_company", "")).upper()

                results.append(
                    {
                        "price": np.log(price),  # Log-transformed
                        "date": sale_date,
                        "days_ago": days_ago,
                        "half_grade": 1.0 if (parsed_grade - grade_floor) > 0 else 0.0,
                        "grade_co_BGS": 1 if grading_co == "BGS" else 0,
                        "grade_co_CGC": 1 if grading_co == "CGC" else 0,
                        "grade_co_PSA": 1 if grading_co == "PSA" else 0,
                    }
                )

                if len(results) >= n_sales:
                    break

        except Exception as e:
            print(f"Error querying eBay sales: {e}")

        return results[:n_sales]

    def get_adjacent_grade_sales(
        self,
        gemrate_id: str,
        grade: float,
        direction: str,  # "above" or "below"
        n_sales: int = 5,
        before_date: Optional[datetime] = None,
    ) -> list[dict]:
        """Get sales for adjacent grade (above or below)."""
        if direction == "above":
            target_grade = min(grade + 1, 10)
        else:
            target_grade = max(grade - 1, 1)

        return self.get_previous_sales(gemrate_id, target_grade, n_sales, before_date)

    def get_neighbor_avg_prices(
        self,
        neighbor_ids: list[str],
        grade: float,
        n_sales_per_neighbor: int = 3,
        before_date: Optional[datetime] = None,
    ) -> dict[str, float]:
        """Get average price for each neighbor card."""
        result = {}
        for neighbor_id in neighbor_ids:
            sales = self.get_previous_sales(
                neighbor_id, grade, n_sales_per_neighbor, before_date
            )
            if sales:
                avg_price = np.mean([s["price"] for s in sales])
                result[neighbor_id] = avg_price
        return result


# Global instance
_mongodb_client: Optional[MongoDBClient] = None


def get_mongodb_client() -> MongoDBClient:
    """Get singleton MongoDB client."""
    global _mongodb_client
    if _mongodb_client is None:
        _mongodb_client = MongoDBClient()
    return _mongodb_client
