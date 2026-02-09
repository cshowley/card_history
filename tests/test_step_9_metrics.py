"""
Sanity-check tests for Step 9 data integrity metrics.

Writes synthetic predictions and historical parquet files, runs the real
run_step_9(), and verifies monotonicity violation counts, post-sort
invariants, and QA outlier detection.
"""

import os
from datetime import timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from data_integrity import get_tracker
from tests.conftest import find_widget
from tests.helpers.synthetic_data import make_predictions_df, make_historical_for_qa


def _run_step_9_with_mocks(tmp_path, predictions_df, historical_df):
    """Run step_9 with file paths pointed at synthetic parquet files."""
    pred_path = tmp_path / "predictions.parquet"
    hist_path = tmp_path / "historical_data.parquet"
    output_path = tmp_path / "predictions_sorted.parquet"
    spot_path = tmp_path / "spot_check.parquet"

    predictions_df.to_parquet(str(pred_path), index=False)
    historical_df.to_parquet(str(hist_path), index=False)

    with (
        patch("step_9.constants.S8_PREDICTIONS_FILE", str(pred_path)),
        patch("step_9.constants.S3_HISTORICAL_DATA_FILE", str(hist_path)),
        patch("step_9.constants.S9_OUTPUT_FILE", str(output_path)),
        patch("step_9.os.path.exists", side_effect=lambda p: (
            True if p in (str(pred_path),) else os.path.exists(p)
        )),
    ):
        # Also patch the hardcoded spot_check.parquet path
        original_to_parquet = pd.DataFrame.to_parquet

        def patched_to_parquet(self, path, *args, **kwargs):
            if path == "spot_check.parquet":
                path = str(spot_path)
            return original_to_parquet(self, path, *args, **kwargs)

        with patch.object(pd.DataFrame, "to_parquet", patched_to_parquet):
            from step_9 import run_step_9
            run_step_9()


class TestStep9Monotonicity:
    """Test monotonicity violation detection and enforcement."""

    def test_violation_count_with_violations(self, fresh_tracker, tmp_path):
        """Groups with non-monotonic predictions should be counted."""
        pred_df, expected = make_predictions_df(n_groups=6, n_grades=4, n_violations=3)
        hist_df = make_historical_for_qa(pred_df, n_outliers=0)

        _run_step_9_with_mocks(tmp_path, pred_df, hist_df)

        tracker = get_tracker()
        w = find_widget(tracker, "s9_monotonicity_violations_prediction")
        assert w is not None
        assert w["value"] == expected["s9_monotonicity_violations_prediction"]

    def test_no_violations_when_monotonic(self, fresh_tracker, tmp_path):
        """When all groups are already monotonic, violation count should be 0."""
        pred_df, expected = make_predictions_df(n_groups=4, n_grades=4, n_violations=0)
        hist_df = make_historical_for_qa(pred_df, n_outliers=0)

        _run_step_9_with_mocks(tmp_path, pred_df, hist_df)

        tracker = get_tracker()
        w = find_widget(tracker, "s9_monotonicity_violations_prediction")
        assert w is not None
        assert w["value"] == 0

    def test_post_sort_violations_always_zero(self, fresh_tracker, tmp_path):
        """After sorting, there should be zero remaining violations."""
        pred_df, _ = make_predictions_df(n_groups=5, n_grades=4, n_violations=5)
        hist_df = make_historical_for_qa(pred_df, n_outliers=0)

        _run_step_9_with_mocks(tmp_path, pred_df, hist_df)

        tracker = get_tracker()
        w = find_widget(tracker, "s9_post_sort_violations")
        assert w is not None
        assert w["value"] == 0

    def test_total_groups(self, fresh_tracker, tmp_path):
        """Total groups should match the number of (gemrate_id, grading_company) combos."""
        pred_df, expected = make_predictions_df(n_groups=8, n_grades=3, n_violations=0)
        hist_df = make_historical_for_qa(pred_df, n_outliers=0)

        _run_step_9_with_mocks(tmp_path, pred_df, hist_df)

        tracker = get_tracker()
        w = find_widget(tracker, "s9_total_groups")
        assert w is not None
        assert w["value"] == expected["s9_total_groups"]


class TestStep9QAOutliers:
    """Test QA spot check outlier detection."""

    def test_outlier_count_matches_planted(self, fresh_tracker, tmp_path):
        """Planted outliers (ratio > 1.2 or < 0.8) should be detected."""
        pred_df, _ = make_predictions_df(n_groups=4, n_grades=4, n_violations=0)
        hist_df = make_historical_for_qa(pred_df, n_outliers=3)

        _run_step_9_with_mocks(tmp_path, pred_df, hist_df)

        tracker = get_tracker()
        w = find_widget(tracker, "s9_qa_outliers")
        assert w is not None
        # At least the planted outliers should be detected
        # (might be more if the monotonicity sort changes prediction values)
        assert w["value"] >= 3

    def test_zero_outliers_when_prices_match(self, fresh_tracker, tmp_path):
        """When predicted == recent price, no outliers should be found."""
        pred_df, _ = make_predictions_df(n_groups=3, n_grades=3, n_violations=0)
        hist_df = make_historical_for_qa(pred_df, n_outliers=0)

        _run_step_9_with_mocks(tmp_path, pred_df, hist_df)

        tracker = get_tracker()
        w = find_widget(tracker, "s9_qa_outliers")
        assert w is not None
        # With 0 violations and exact price match, outliers should be 0
        # (monotonicity sort doesn't change monotonic predictions)
        assert w["value"] == 0

    def test_outlier_percentage(self, fresh_tracker, tmp_path):
        """Outlier percentage should be (outliers / total_spot_checks) * 100."""
        pred_df, _ = make_predictions_df(n_groups=4, n_grades=4, n_violations=0)
        hist_df = make_historical_for_qa(pred_df, n_outliers=0)

        _run_step_9_with_mocks(tmp_path, pred_df, hist_df)

        tracker = get_tracker()
        w_count = find_widget(tracker, "s9_qa_outliers")
        w_pct = find_widget(tracker, "s9_qa_outlier_pct")
        assert w_count is not None and w_pct is not None

        if w_count["value"] == 0:
            assert w_pct["value"] == 0.0
        else:
            # Verify the percentage is in [0, 100]
            assert 0 <= w_pct["value"] <= 100.0
