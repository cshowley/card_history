"""
Sanity-check tests for Step 1 data integrity metrics.

Mocks MongoDB and HTTP to inject synthetic DataFrames, then verifies
that all metric computations in run_step_1() produce expected values.
"""

import os
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from data_integrity import get_tracker
from tests.conftest import find_widget
from tests.helpers.synthetic_data import make_ebay_df, make_pwcc_df


def _build_mock_mongo(ebay_df, pwcc_df):
    """Create a mock MongoClient that returns our synthetic DataFrames.

    run_step_1 does:
        client = MongoClient(url)
        db = client[db_name]
        ebay_collection = db[collection_name]
        ebay_results = ebay_collection.aggregate(...)
        ebay_df = pd.json_normalize(list(ebay_results))

    So we need aggregate() to return an iterable of dicts.
    """
    mock_client = MagicMock()
    mock_db = MagicMock()
    mock_client.__getitem__ = MagicMock(return_value=mock_db)

    ebay_records = ebay_df.to_dict(orient="records")
    pwcc_records = pwcc_df.to_dict(orient="records")

    mock_ebay_collection = MagicMock()
    mock_ebay_collection.aggregate.return_value = ebay_records
    mock_pwcc_collection = MagicMock()
    mock_pwcc_collection.aggregate.return_value = pwcc_records

    call_count = {"n": 0}
    collections = {"ebay_graded_items": mock_ebay_collection, "pwcc_graded_items": mock_pwcc_collection}

    def getitem(name):
        return collections.get(name, MagicMock())

    mock_db.__getitem__ = MagicMock(side_effect=getitem)

    return mock_client


def _run_step_1_with_mocks(ebay_df, pwcc_df, tmp_path):
    """Run step_1 with all external I/O mocked."""
    mock_client = _build_mock_mongo(ebay_df, pwcc_df)

    index_json = [
        {"date": "2025-10-01", "value": 100.0},
        {"date": "2025-10-02", "value": 101.0},
    ]

    mock_response = MagicMock()
    mock_response.json.return_value = index_json
    mock_response.raise_for_status.return_value = None

    with (
        patch("step_1.MongoClient", return_value=mock_client),
        patch("step_1.requests.get", return_value=mock_response),
        patch("step_1.constants.S1_MONGO_URL", "mongodb://fake"),
        patch("step_1.constants.S1_EBAY_MARKET_FILE", str(tmp_path / "ebay.csv")),
        patch("step_1.constants.S1_PWCC_MARKET_FILE", str(tmp_path / "pwcc.csv")),
        patch("step_1.constants.S1_INDEX_FILE", str(tmp_path / "index.csv")),
    ):
        from step_1 import run_step_1
        run_step_1()


class TestStep1RecordCounts:
    """Test that row counts and cutoff filtering are accurate."""

    def test_total_and_source_counts(self, fresh_tracker, tmp_path):
        ebay_df, ebay_exp = make_ebay_df(n_rows=10, n_before_cutoff=3)
        pwcc_df, pwcc_exp = make_pwcc_df(n_rows=5, n_before_cutoff=1)

        _run_step_1_with_mocks(ebay_df, pwcc_df, tmp_path)

        tracker = get_tracker()

        ebay_count = find_widget(tracker, "s1_ebay_records")
        assert ebay_count is not None
        assert ebay_count["value"] == ebay_exp["s1_ebay_records"]

        pwcc_count = find_widget(tracker, "s1_pwcc_records")
        assert pwcc_count is not None
        assert pwcc_count["value"] == pwcc_exp["s1_pwcc_records"]

        total = find_widget(tracker, "s1_total_records")
        assert total is not None
        assert total["value"] == ebay_exp["s1_ebay_records"] + pwcc_exp["s1_pwcc_records"]

    def test_sales_before_cutoff(self, fresh_tracker, tmp_path):
        ebay_df, _ = make_ebay_df(n_rows=10, n_before_cutoff=4)
        pwcc_df, _ = make_pwcc_df(n_rows=5, n_before_cutoff=2)

        _run_step_1_with_mocks(ebay_df, pwcc_df, tmp_path)

        tracker = get_tracker()
        w = find_widget(tracker, "s1_sales_before_sep_2025")
        assert w is not None
        assert w["value"] == 6  # 4 ebay + 2 pwcc

    def test_dropped_sales_count(self, fresh_tracker, tmp_path):
        ebay_df, _ = make_ebay_df(n_rows=8, n_before_cutoff=2)
        pwcc_df, _ = make_pwcc_df(n_rows=4, n_before_cutoff=1)

        _run_step_1_with_mocks(ebay_df, pwcc_df, tmp_path)

        tracker = get_tracker()
        w = find_widget(tracker, "s1_dropped_sales_before_sep_2025")
        assert w is not None
        assert w["value"] == 3  # 2 ebay + 1 pwcc

    def test_zero_before_cutoff(self, fresh_tracker, tmp_path):
        """When no sales are before the cutoff, dropped count should be 0."""
        ebay_df, _ = make_ebay_df(n_rows=5, n_before_cutoff=0)
        pwcc_df, _ = make_pwcc_df(n_rows=3, n_before_cutoff=0)

        _run_step_1_with_mocks(ebay_df, pwcc_df, tmp_path)

        tracker = get_tracker()
        w = find_widget(tracker, "s1_sales_before_sep_2025")
        assert w["value"] == 0
        wd = find_widget(tracker, "s1_dropped_sales_before_sep_2025")
        assert wd["value"] == 0


class TestStep1AnomalyDetection:
    """Test bid anomaly and price outlier metrics."""

    def test_bid_anomalies(self, fresh_tracker, tmp_path):
        ebay_df, _ = make_ebay_df(
            n_rows=10, n_before_cutoff=0,
            n_zero_bids=3, n_single_bids=2,
        )
        pwcc_df, _ = make_pwcc_df(n_rows=3, n_before_cutoff=0)

        _run_step_1_with_mocks(ebay_df, pwcc_df, tmp_path)

        tracker = get_tracker()
        table = find_widget(tracker, "s1_anomalies")
        assert table is not None

        # Table data: [[\"Zero-bid auctions\", \"3\"], [\"Single-bid auctions\", \"2\"]]
        assert table["data"][0][1] == "3"
        assert table["data"][1][1] == "2"

    def test_extreme_prices(self, fresh_tracker, tmp_path):
        """Extreme prices are counted AFTER cutoff filtering."""
        # All rows are after cutoff (n_before_cutoff=0), so extremes survive
        ebay_df, _ = make_ebay_df(
            n_rows=8, n_before_cutoff=0,
            n_extreme_low_price=2, n_extreme_high_price=1,
            n_missing_price=0,
        )
        pwcc_df, _ = make_pwcc_df(
            n_rows=4, n_before_cutoff=0,
            n_extreme_low_price=1, n_extreme_high_price=1,
        )

        _run_step_1_with_mocks(ebay_df, pwcc_df, tmp_path)

        tracker = get_tracker()
        w = find_widget(tracker, "s1_extreme_prices_count")
        assert w is not None
        # ebay: 2 low + 1 high = 3, pwcc: 1 low + 1 high = 2, total = 5
        assert w["value"] == 5


class TestStep1MissingData:
    """Test missing data analysis tables."""

    def test_ebay_missing_gemrate_id(self, fresh_tracker, tmp_path):
        ebay_df, _ = make_ebay_df(
            n_rows=10, n_before_cutoff=0,
            n_missing_gemrate_id=2, n_missing_grade=0, n_missing_price=0,
        )
        pwcc_df, _ = make_pwcc_df(n_rows=2, n_before_cutoff=0, n_missing_gemrate_id=0)

        _run_step_1_with_mocks(ebay_df, pwcc_df, tmp_path)

        tracker = get_tracker()
        table = find_widget(tracker, "s1_ebay_missing")
        assert table is not None
        # First row should be gemrate_id with count "2"
        gid_row = [r for r in table["data"] if r[0] == "gemrate_id"]
        assert len(gid_row) == 1
        assert gid_row[0][1] == "2"

    def test_pwcc_missing_data(self, fresh_tracker, tmp_path):
        ebay_df, _ = make_ebay_df(n_rows=5, n_before_cutoff=0, n_missing_gemrate_id=0)
        pwcc_df, _ = make_pwcc_df(
            n_rows=6, n_before_cutoff=0,
            n_missing_gemrate_id=3, n_missing_grade=0,
        )

        _run_step_1_with_mocks(ebay_df, pwcc_df, tmp_path)

        tracker = get_tracker()
        table = find_widget(tracker, "s1_pwcc_missing")
        assert table is not None
        gid_row = [r for r in table["data"] if r[0] == "gemrate_id"]
        assert len(gid_row) == 1
        assert gid_row[0][1] == "3"


class TestStep1MarketplaceBreakdown:
    """Test the marketplace share table."""

    def test_share_percentages(self, fresh_tracker, tmp_path):
        ebay_df, _ = make_ebay_df(n_rows=8, n_before_cutoff=0)
        pwcc_df, _ = make_pwcc_df(n_rows=2, n_before_cutoff=0)

        _run_step_1_with_mocks(ebay_df, pwcc_df, tmp_path)

        tracker = get_tracker()
        table = find_widget(tracker, "s1_marketplace_breakdown")
        assert table is not None
        # eBay: 8/10 = 80%, PWCC: 2/10 = 20%
        ebay_row = table["data"][0]
        pwcc_row = table["data"][1]
        assert ebay_row[0] == "eBay"
        assert ebay_row[2] == 80.0
        assert pwcc_row[0] == "PWCC"
        assert pwcc_row[2] == 20.0


class TestStep1GradeDistribution:
    """Test grade distribution charts."""

    def test_ebay_grade_chart_emitted(self, fresh_tracker, tmp_path):
        ebay_df, _ = make_ebay_df(n_rows=6, n_before_cutoff=0)
        pwcc_df, _ = make_pwcc_df(n_rows=2, n_before_cutoff=0)

        _run_step_1_with_mocks(ebay_df, pwcc_df, tmp_path)

        tracker = get_tracker()
        chart = find_widget(tracker, "s1_ebay_grades")
        assert chart is not None
        assert chart["type"] == "chart"
        # Should have at least one data series
        assert len(chart["data"]) > 0
