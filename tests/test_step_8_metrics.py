"""
Sanity-check tests for Step 8 data integrity metrics.

Tests prediction distribution statistics (min, max, mean, median, std,
negative count) and model collapse detection by feeding known prediction
arrays through the same computation that step_8 performs.
"""

import numpy as np
import pandas as pd
import pytest

from data_integrity import get_tracker
from tests.conftest import find_widget


def _simulate_step8_metrics(tracker, predictions, predictions_lower=None, predictions_upper=None):
    """Replicate the exact metric computation from step_8.py lines 176-251.

    This mirrors the production code so tests verify the formula accuracy.
    """
    total_rows = len(predictions)
    tracker.add_metric(
        id="s8_predictions_generated",
        title="Predictions Generated",
        value=total_rows,
    )

    negative_predictions_count = int(np.sum(np.array(predictions) < 0))

    if predictions:
        preds_arr = np.array(predictions)
        pred_min = float(np.min(preds_arr))
        pred_max = float(np.max(preds_arr))
        pred_mean = float(np.mean(preds_arr))
        pred_median = float(np.median(preds_arr))
        pred_std = float(np.std(preds_arr))

        tracker.add_metric(id="s8_prediction_min", title="Prediction Min", value=round(pred_min, 4))
        tracker.add_metric(id="s8_prediction_max", title="Prediction Max", value=round(pred_max, 4))
        tracker.add_metric(id="s8_prediction_mean", title="Prediction Mean", value=round(pred_mean, 4))
        tracker.add_metric(id="s8_prediction_median", title="Prediction Median", value=round(pred_median, 4))
        tracker.add_metric(id="s8_prediction_std", title="Prediction Std Dev", value=round(pred_std, 4))
        tracker.add_metric(
            id="s8_negative_predictions_count",
            title="Negative Predictions Count",
            value=negative_predictions_count,
        )

        if pred_min == pred_max:
            tracker.add_error(
                f"Model collapse detected: all predictions are {pred_min}",
                step="step_8",
            )

    if predictions_lower:
        lower_arr = np.array(predictions_lower)
        tracker.add_metric(
            id="s8_prediction_lower_min", title="Prediction Lower Min",
            value=round(float(np.min(lower_arr)), 4),
        )
        tracker.add_metric(
            id="s8_prediction_lower_max", title="Prediction Lower Max",
            value=round(float(np.max(lower_arr)), 4),
        )

    if predictions_upper:
        upper_arr = np.array(predictions_upper)
        tracker.add_metric(
            id="s8_prediction_upper_min", title="Prediction Upper Min",
            value=round(float(np.min(upper_arr)), 4),
        )
        tracker.add_metric(
            id="s8_prediction_upper_max", title="Prediction Upper Max",
            value=round(float(np.max(upper_arr)), 4),
        )


class TestStep8PredictionStats:
    """Test prediction distribution statistics with known arrays."""

    def test_basic_stats(self, fresh_tracker):
        predictions = [10.0, 20.0, 30.0, 40.0, 50.0]

        _simulate_step8_metrics(fresh_tracker, predictions)

        assert find_widget(fresh_tracker, "s8_predictions_generated")["value"] == 5
        assert find_widget(fresh_tracker, "s8_prediction_min")["value"] == 10.0
        assert find_widget(fresh_tracker, "s8_prediction_max")["value"] == 50.0
        assert find_widget(fresh_tracker, "s8_prediction_mean")["value"] == 30.0
        assert find_widget(fresh_tracker, "s8_prediction_median")["value"] == 30.0

        # std of [10,20,30,40,50] = sqrt(200) = 14.1421...
        expected_std = round(float(np.std([10, 20, 30, 40, 50])), 4)
        assert find_widget(fresh_tracker, "s8_prediction_std")["value"] == expected_std

    def test_negative_predictions_count(self, fresh_tracker):
        predictions = [10.0, -5.0, 20.0, -3.0, -1.0, 50.0]

        _simulate_step8_metrics(fresh_tracker, predictions)

        w = find_widget(fresh_tracker, "s8_negative_predictions_count")
        assert w is not None
        assert w["value"] == 3

    def test_no_negative_predictions(self, fresh_tracker):
        predictions = [10.0, 20.0, 30.0]

        _simulate_step8_metrics(fresh_tracker, predictions)

        w = find_widget(fresh_tracker, "s8_negative_predictions_count")
        assert w["value"] == 0

    def test_single_prediction(self, fresh_tracker):
        predictions = [42.0]

        _simulate_step8_metrics(fresh_tracker, predictions)

        assert find_widget(fresh_tracker, "s8_prediction_min")["value"] == 42.0
        assert find_widget(fresh_tracker, "s8_prediction_max")["value"] == 42.0
        assert find_widget(fresh_tracker, "s8_prediction_mean")["value"] == 42.0
        assert find_widget(fresh_tracker, "s8_prediction_median")["value"] == 42.0
        assert find_widget(fresh_tracker, "s8_prediction_std")["value"] == 0.0


class TestStep8ModelCollapse:
    """Test model collapse detection."""

    def test_collapse_detected_when_all_same(self, fresh_tracker):
        predictions = [25.0, 25.0, 25.0, 25.0]

        _simulate_step8_metrics(fresh_tracker, predictions)

        errors = fresh_tracker.get_data()["errors"]
        assert len(errors) == 1
        assert "Model collapse" in errors[0]
        assert "25.0" in errors[0]

    def test_no_collapse_when_varied(self, fresh_tracker):
        predictions = [10.0, 20.0, 30.0]

        _simulate_step8_metrics(fresh_tracker, predictions)

        errors = fresh_tracker.get_data()["errors"]
        assert len(errors) == 0


class TestStep8BoundMetrics:
    """Test lower/upper prediction bound metrics."""

    def test_lower_bound_stats(self, fresh_tracker):
        preds = [10.0, 20.0, 30.0]
        lower = [5.0, 15.0, 25.0]

        _simulate_step8_metrics(fresh_tracker, preds, predictions_lower=lower)

        assert find_widget(fresh_tracker, "s8_prediction_lower_min")["value"] == 5.0
        assert find_widget(fresh_tracker, "s8_prediction_lower_max")["value"] == 25.0

    def test_upper_bound_stats(self, fresh_tracker):
        preds = [10.0, 20.0, 30.0]
        upper = [15.0, 25.0, 35.0]

        _simulate_step8_metrics(fresh_tracker, preds, predictions_upper=upper)

        assert find_widget(fresh_tracker, "s8_prediction_upper_min")["value"] == 15.0
        assert find_widget(fresh_tracker, "s8_prediction_upper_max")["value"] == 35.0

    def test_all_bounds_together(self, fresh_tracker):
        preds = [100.0, 200.0]
        lower = [80.0, 160.0]
        upper = [120.0, 240.0]

        _simulate_step8_metrics(fresh_tracker, preds, lower, upper)

        assert find_widget(fresh_tracker, "s8_prediction_min")["value"] == 100.0
        assert find_widget(fresh_tracker, "s8_prediction_max")["value"] == 200.0
        assert find_widget(fresh_tracker, "s8_prediction_lower_min")["value"] == 80.0
        assert find_widget(fresh_tracker, "s8_prediction_lower_max")["value"] == 160.0
        assert find_widget(fresh_tracker, "s8_prediction_upper_min")["value"] == 120.0
        assert find_widget(fresh_tracker, "s8_prediction_upper_max")["value"] == 240.0
