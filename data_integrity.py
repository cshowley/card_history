"""
Data Integrity Tracker Module

Provides a global tracker to collect data integrity metrics throughout
the pipeline execution. Metrics are stored as widgets (metric, chart, table)
and saved to MongoDB at the end of the run.

Schema Design (simplified, ContEx-inspired):
- Metric: {id, type, title, value}
- Chart: {id, type, title, chart_type, columns, data}
- Table: {id, type, title, columns, data}

Chart and Table share the same Dataset structure (columns + data rows).
"""

import os
from datetime import datetime, timezone
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# Database configuration
DATABASE = "gemrate"
COLLECTION = "data_integrity"


class DataIntegrityTracker:
    """
    Tracks data integrity metrics throughout pipeline execution.
    Supports three widget types: metric, chart, and table.
    """

    def __init__(
        self, title: str = "Card History Pipeline Metrics", version: str = "1.0"
    ):
        self.data = {
            "meta": {"title": title, "last_updated": None, "version": version},
            "errors": [],
            "widgets": [],
        }

    def add_metric(self, id: str, title: str, value: str):
        """Add a single metric widget."""
        widget = {
            "id": id,
            "type": "metric",
            "title": title,
            "value": value,
        }
        self.data["widgets"].append(widget)

    def add_chart(
        self,
        id: str,
        title: str,
        chart_type: str,
        columns: list,
        data: list,
    ):
        """
        Add a chart widget using Dataset format.

        Args:
            id: Unique identifier for the widget
            title: Display title
            chart_type: Type of chart (e.g., "line", "bar", "pie")
            columns: List of column names (e.g., ["date", "sales", "volume"])
            data: List of rows, each row is a list matching columns
        """
        widget = {
            "id": id,
            "type": "chart",
            "title": title,
            "chart_type": chart_type,
            "columns": columns,
            "data": data,
        }
        self.data["widgets"].append(widget)

    def add_table(self, id: str, title: str, columns: list, data: list):
        """
        Add a table widget using Dataset format.

        Args:
            id: Unique identifier for the widget
            title: Display title
            columns: List of column headers
            data: List of rows, each row is a list matching columns
        """
        widget = {
            "id": id,
            "type": "table",
            "title": title,
            "columns": columns,
            "data": data,
        }
        self.data["widgets"].append(widget)

    def add_error(self, error_message: str, step: str = None):
        """Add an error message."""
        error_entry = error_message
        if step:
            error_entry = f"[{step}] {error_message}"
        self.data["errors"].append(error_entry)

    def get_data(self) -> dict:
        """Return the current data integrity JSON."""
        return self.data


# Global tracker instance
_tracker = None


def get_tracker() -> DataIntegrityTracker:
    """Get the global DataIntegrityTracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = DataIntegrityTracker()
    return _tracker


def reset_tracker():
    """Reset the global tracker (useful for testing)."""
    global _tracker
    _tracker = None


def save_to_mongo():
    """
    Save the data integrity JSON to MongoDB.

    Uses MONGO_URI environment variable for connection.
    Saves to database 'gemrate', collection 'data_integrity'.
    """
    tracker = get_tracker()

    # Set the timestamp
    tracker.data["meta"]["last_updated"] = datetime.now(timezone.utc).isoformat()

    # Get MongoDB connection
    mongo_uri = os.getenv("MONGO_URI_RW")
    if not mongo_uri:
        raise ValueError("MONGO_URI_RW not found in environment")

    print("Connecting to MongoDB for data integrity save...")
    client = MongoClient(mongo_uri)
    db = client[DATABASE]
    collection = db[COLLECTION]

    # Use a fixed document ID so we always update the same document
    document_id = "pipeline_run_latest"

    # Prepare the document
    doc = tracker.get_data()
    doc["_id"] = document_id

    # Upsert the document
    result = collection.update_one({"_id": document_id}, {"$set": doc}, upsert=True)

    if result.upserted_id:
        print(f"Data integrity document created in '{COLLECTION}' collection.")
    else:
        print(f"Data integrity document updated in '{COLLECTION}' collection.")

    client.close()

    return result
