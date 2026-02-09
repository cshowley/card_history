"""
Sanity-check tests for Step 3 data integrity chart/table metrics.

Step 3 is the most complex step to mock (6+ file reads, batch Parquet writes,
feature engineering). Rather than mocking the entire run_step_3(), we replicate
the metric computation logic from lines 424-577 against a controlled DataFrame
and verify the tracker produces identical results.

This tests the *accuracy of the metric computations themselves* -- confirming
that the pandas groupby/agg logic produces correct values for known data.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from data_integrity import get_tracker
from tests.conftest import find_widget


def _build_sales_df(n_days=28, sales_per_day=5, base_price=50.0):
    """Build a synthetic sales DataFrame matching step 3's expected schema.

    All dates are within the last ``n_days`` days so they fall inside the
    28-day cutoff window used by step 3's metrics.
    """
    rng = np.random.default_rng(42)
    today = pd.Timestamp.now().normalize()
    rows = []

    for day_offset in range(n_days):
        sale_date = today - pd.Timedelta(days=n_days - 1 - day_offset)
        for s in range(sales_per_day):
            price = base_price + rng.uniform(-20, 80)
            grade = rng.choice([7, 8, 9, 10])
            rows.append({
                "gemrate_id": f"GEM_{s % 10:04d}",
                "grade": float(grade),
                "half_grade": 0.0,
                "date": sale_date,
                "price": round(price, 2),
                "grading_company": rng.choice(["PSA", "BGS", "CGC"]),
                "seller": f"seller_{s % 3}",
            })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _compute_expected_sales_per_day(df, n_days=28):
    """Replicate step 3's sales_per_day computation."""
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=n_days)
    recent = df[df["date"] >= cutoff]
    return recent.groupby(recent["date"].dt.date).size().sort_index()


def _compute_expected_dollar_volume(df, n_days=28):
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=n_days)
    recent = df[df["date"] >= cutoff]
    return recent.groupby(recent["date"].dt.date)["price"].sum().sort_index()


def _compute_expected_median_price(df, n_days=28):
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=n_days)
    recent = df[df["date"] >= cutoff]
    return recent.groupby(recent["date"].dt.date)["price"].median().sort_index()


class TestStep3SalesCharts:
    """Verify chart data matches independent computation on the same data."""

    def test_sales_per_day_chart(self, fresh_tracker):
        df = _build_sales_df(n_days=10, sales_per_day=3)
        expected_spd = _compute_expected_sales_per_day(df, n_days=28)

        # Replicate what step 3 does: compute and push to tracker
        tracker = get_tracker()
        cutoff_date = pd.Timestamp.now().normalize() - pd.Timedelta(days=28)
        recent_sales = df[df["date"] >= cutoff_date]
        sales_per_day = recent_sales.groupby(recent_sales["date"].dt.date).size().sort_index()
        chart_data = [[str(d), int(count)] for d, count in sales_per_day.items()]

        tracker.add_chart(
            id="sales_per_day",
            title="Sales Per Day",
            chart_type="line",
            columns=["date", "sales"],
            data=chart_data,
        )

        chart = find_widget(tracker, "sales_per_day")
        assert chart is not None
        assert len(chart["data"]) == len(expected_spd)

        # Verify each day's count
        for row, (exp_date, exp_count) in zip(chart["data"], expected_spd.items()):
            assert row[0] == str(exp_date)
            assert row[1] == int(exp_count)

    def test_dollar_volume_chart(self, fresh_tracker):
        df = _build_sales_df(n_days=7, sales_per_day=4)
        expected_dv = _compute_expected_dollar_volume(df, n_days=28)

        tracker = get_tracker()
        cutoff_date = pd.Timestamp.now().normalize() - pd.Timedelta(days=28)
        recent_sales = df[df["date"] >= cutoff_date]
        dollar_vol = recent_sales.groupby(recent_sales["date"].dt.date)["price"].sum().sort_index()
        dollar_chart_data = [
            [str(d), round(float(v), 2)] for d, v in dollar_vol.items()
        ]

        tracker.add_chart(
            id="dollar_volume_per_day",
            title="Dollar Volume Per Day",
            chart_type="line",
            columns=["date", "dollar_volume"],
            data=dollar_chart_data,
        )

        chart = find_widget(tracker, "dollar_volume_per_day")
        assert chart is not None

        for row, (exp_date, exp_vol) in zip(chart["data"], expected_dv.items()):
            assert row[0] == str(exp_date)
            assert row[1] == round(float(exp_vol), 2)

    def test_median_price_chart(self, fresh_tracker):
        df = _build_sales_df(n_days=5, sales_per_day=6)
        expected_mp = _compute_expected_median_price(df, n_days=28)

        tracker = get_tracker()
        cutoff_date = pd.Timestamp.now().normalize() - pd.Timedelta(days=28)
        recent_sales = df[df["date"] >= cutoff_date]
        median_p = recent_sales.groupby(recent_sales["date"].dt.date)["price"].median().sort_index()
        chart_data = [
            [str(d), round(float(p), 2)] for d, p in median_p.items()
        ]

        tracker.add_chart(
            id="median_price_per_day",
            title="Median Sales Price Per Day",
            chart_type="line",
            columns=["date", "median_price"],
            data=chart_data,
        )

        chart = find_widget(tracker, "median_price_per_day")
        assert chart is not None

        for row, (exp_date, exp_price) in zip(chart["data"], expected_mp.items()):
            assert row[0] == str(exp_date)
            assert row[1] == round(float(exp_price), 2)


class TestStep3PriceHistogram:
    """Test the log-bin price histogram for the most recent day."""

    def test_histogram_bins(self, fresh_tracker):
        # Create sales on one day with known price distribution
        today = pd.Timestamp.now().normalize()
        prices = [0.50, 5.0, 50.0, 500.0, 5000.0, 50000.0]
        df = pd.DataFrame({
            "gemrate_id": [f"GEM_{i:04d}" for i in range(len(prices))],
            "grade": [9.0] * len(prices),
            "date": [today] * len(prices),
            "price": prices,
        })

        bins = [0, 1, 10, 100, 1000, 10000, float("inf")]
        bin_labels = ["$0-1", "$1-10", "$10-100", "$100-1K", "$1K-10K", "$10K+"]
        today_prices = df["price"].dropna()
        bin_counts = pd.cut(today_prices, bins=bins, labels=bin_labels).value_counts()
        bin_counts = bin_counts.reindex(bin_labels, fill_value=0)

        tracker = get_tracker()
        histogram_data = [[label, int(count)] for label, count in bin_counts.items()]
        tracker.add_chart(
            id="sales_price_histogram",
            title=f"Sales Price Distribution ({today.date()})",
            chart_type="bar",
            columns=["price_range", "count"],
            data=histogram_data,
        )

        chart = find_widget(tracker, "sales_price_histogram")
        assert chart is not None
        # Each bin should have exactly 1 item
        for row in chart["data"]:
            assert row[1] == 1, f"Bin {row[0]} expected 1, got {row[1]}"


class TestStep3MedianDaysBetweenSales:
    """Test the median days between sales for top cards."""

    def test_median_days_computation(self, fresh_tracker):
        # Card A: sales every 7 days => median = 7
        # Card B: sales every 14 days => median = 14
        # Overall median of [7, 7, 14, 14] = 10.5
        today = pd.Timestamp.now().normalize()
        rows = []
        for i in range(3):
            rows.append({"gemrate_id": "A", "date": today - pd.Timedelta(days=i * 7), "price": 10.0, "grade": 9.0})
        for i in range(3):
            rows.append({"gemrate_id": "B", "date": today - pd.Timedelta(days=i * 14), "price": 20.0, "grade": 10.0})

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])

        # Replicate step 3 computation
        top_ids = df["gemrate_id"].value_counts().head(100).index.tolist()
        top_df = df[df["gemrate_id"].isin(top_ids)].copy()
        top_df = top_df.sort_values(["gemrate_id", "date"]).reset_index(drop=True)
        top_df["prev_sale_date"] = top_df.groupby("gemrate_id")["date"].shift(1)
        top_df["days_between"] = (top_df["date"] - top_df["prev_sale_date"]).dt.days
        days_between_all = top_df["days_between"].dropna()
        median_days = days_between_all.median()

        tracker = get_tracker()
        tracker.add_metric(
            id="s3_median_days_between_top_100_sales",
            title="Median Days Between Sales (Top 100 Cards)",
            value=round(median_days, 1),
        )

        w = find_widget(tracker, "s3_median_days_between_top_100_sales")
        assert w is not None
        # A: intervals [7, 7], B: intervals [14, 14], median of [7, 7, 14, 14] = 10.5
        assert w["value"] == 10.5


class TestStep3SalesConcentration:
    """Test the sales concentration table computation."""

    def test_concentration_data(self, fresh_tracker):
        today = pd.Timestamp.now().normalize()
        yesterday = today - pd.Timedelta(days=1)
        df = pd.DataFrame({
            "gemrate_id": ["A", "A", "B", "C", "C"],
            "date": [today, today, today, yesterday, yesterday],
            "price": [10, 20, 30, 40, 50],
            "grade": [9.0] * 5,
        })
        df["date"] = pd.to_datetime(df["date"])

        cutoff_date = pd.Timestamp.now().normalize() - pd.Timedelta(days=28)
        recent_sales = df[df["date"] >= cutoff_date]
        sales_per_day = recent_sales.groupby(recent_sales["date"].dt.date).size().sort_index()

        tracker = get_tracker()
        concentration_data = []
        for sale_date in sorted(sales_per_day.index):
            day_sales = recent_sales[recent_sales["date"].dt.date == sale_date]
            total_sales = len(day_sales)
            unique_cards = day_sales["gemrate_id"].nunique()
            concentration_data.append([str(sale_date), total_sales, unique_cards])

        tracker.add_table(
            id="sales_concentration_per_day",
            title="Sales Concentration Per Day",
            columns=["date", "total_sales", "unique_cards"],
            data=concentration_data,
        )

        table = find_widget(tracker, "sales_concentration_per_day")
        assert table is not None
        assert len(table["data"]) == 2  # two days

        # yesterday: 2 sales, 1 unique card (C)
        yesterday_row = table["data"][0]
        assert yesterday_row[1] == 2
        assert yesterday_row[2] == 1

        # today: 3 sales, 2 unique cards (A, B)
        today_row = table["data"][1]
        assert today_row[1] == 3
        assert today_row[2] == 2
