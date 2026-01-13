import glob
import json
import os
import platform

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from xgboost import XGBRegressor


# Quantile configuration for prediction intervals
LOWER_QUANTILE = 0.025  # 2.5th percentile
UPPER_QUANTILE = (
    0.985  # 98.5th percentile (adjusted from 0.975 to improve upper bound coverage)
)
TARGET_COVERAGE = 0.95  # Expected coverage (95%)


def get_xgb_device():
    """Determine best device for XGBoost: CUDA -> MPS (CPU) -> CPU."""
    # Try CUDA first (NVIDIA GPU)
    try:
        import cupy as cp

        cp.cuda.Device(0).compute_capability
        print("Using CUDA (NVIDIA GPU)")
        return "cuda:0"
    except Exception:
        pass

    # Check for Apple Silicon (MPS)
    # Note: XGBoost doesn't have native MPS support, but on Apple Silicon
    # it uses CPU with Apple Accelerate framework for optimization
    if platform.processor() == "arm" or "Apple" in platform.processor():
        print("Using CPU with Apple Accelerate (M-series optimized)")
        return "cpu"

    print("Using CPU")
    return "cpu"


def load_data(data_dir):
    print(f"Loading data from {data_dir}...")
    X_train = pd.read_pickle(os.path.join(data_dir, "X_train.pkl"))
    y_train = pd.read_pickle(os.path.join(data_dir, "y_train.pkl"))
    X_val = pd.read_pickle(os.path.join(data_dir, "X_val.pkl"))
    y_val = pd.read_pickle(os.path.join(data_dir, "y_val.pkl"))
    X_test = pd.read_pickle(os.path.join(data_dir, "X_test.pkl"))
    y_test = pd.read_pickle(os.path.join(data_dir, "y_test.pkl"))
    return X_train, y_train, X_val, y_val, X_test, y_test


def find_best_grid(results_dir):
    print(f"Scanning {results_dir} for results...")
    result_files = glob.glob(os.path.join(results_dir, "worker_*_best.json"))

    global_best_mape = float("inf")
    global_best_params = {}

    if not result_files:
        print("No result files found in model/results/.")
        return None

    for f in result_files:
        try:
            with open(f, "r") as fp:
                data = json.load(fp)
                mape = data.get("best_mape", float("inf"))
                print(f"File {f}: MAPE = {mape:.5f}")
                if mape < global_best_mape:
                    global_best_mape = mape
                    global_best_params = data.get("best_params", {})
        except Exception as e:
            print(f"Error reading {f}: {e}")

    print(f"Global Best MAPE: {global_best_mape:.5f}")
    print(f"Global Best Params: {global_best_params}")
    return global_best_params


def train_quantile_models(X_train, y_train, best_params, device):
    """Train lower and upper quantile models for prediction intervals."""
    print(
        f"\nTraining quantile models for {TARGET_COVERAGE:.0%} prediction intervals..."
    )

    # Remove objective from best_params if present (we'll set our own)
    quantile_params = {k: v for k, v in best_params.items() if k != "objective"}

    # Train lower bound model (2.5th percentile)
    print(f"  Training lower bound model (α={LOWER_QUANTILE})...")
    model_lower = XGBRegressor(
        device=device,
        objective="reg:quantileerror",
        quantile_alpha=LOWER_QUANTILE,
        **quantile_params,
    )
    model_lower.fit(X_train, y_train, verbose=False)

    # Train upper bound model (97.5th percentile)
    print(f"  Training upper bound model (α={UPPER_QUANTILE})...")
    model_upper = XGBRegressor(
        device=device,
        objective="reg:quantileerror",
        quantile_alpha=UPPER_QUANTILE,
        **quantile_params,
    )
    model_upper.fit(X_train, y_train, verbose=False)

    return model_lower, model_upper


def evaluate_coverage(y_true, y_pred_lower, y_pred_upper, space_name="log"):
    """
    Evaluate prediction interval coverage and calibration.

    Returns dict with:
    - coverage: % of samples within interval
    - below_lower: % of samples below lower bound (should be ~2.5%)
    - above_upper: % of samples above upper bound (should be ~2.5%)
    - avg_interval_width: average width of intervals
    - median_interval_width: median width of intervals
    """
    n_samples = len(y_true)

    below_lower = np.sum(y_true < y_pred_lower)
    above_upper = np.sum(y_true > y_pred_upper)
    within_interval = np.sum((y_true >= y_pred_lower) & (y_true <= y_pred_upper))

    interval_widths = y_pred_upper - y_pred_lower

    results = {
        "space": space_name,
        "coverage": float(within_interval / n_samples),
        "below_lower": float(below_lower / n_samples),
        "above_upper": float(above_upper / n_samples),
        "avg_interval_width": float(np.mean(interval_widths)),
        "median_interval_width": float(np.median(interval_widths)),
        "min_interval_width": float(np.min(interval_widths)),
        "max_interval_width": float(np.max(interval_widths)),
    }

    return results


def print_coverage_report(coverage_results, target_coverage=TARGET_COVERAGE):
    """Print a formatted coverage report with pass/fail indicators."""
    print("\n" + "=" * 60)
    print(f"PREDICTION INTERVAL COVERAGE REPORT ({coverage_results['space']} space)")
    print("=" * 60)

    # Coverage check
    actual_coverage = coverage_results["coverage"]
    coverage_tolerance = 0.03  # Allow 3% deviation
    coverage_ok = abs(actual_coverage - target_coverage) <= coverage_tolerance
    status = "✓ PASS" if coverage_ok else "✗ FAIL"
    print(
        f"\nOverall Coverage: {actual_coverage:.1%} (target: {target_coverage:.0%}) {status}"
    )

    # Lower bound calibration
    expected_below = (1 - target_coverage) / 2
    actual_below = coverage_results["below_lower"]
    lower_tolerance = 0.02
    lower_ok = abs(actual_below - expected_below) <= lower_tolerance
    status = "✓" if lower_ok else "✗"
    print(
        f"Below Lower Bound: {actual_below:.1%} (expected: {expected_below:.1%}) {status}"
    )

    # Upper bound calibration
    expected_above = (1 - target_coverage) / 2
    actual_above = coverage_results["above_upper"]
    upper_ok = abs(actual_above - expected_above) <= lower_tolerance
    status = "✓" if upper_ok else "✗"
    print(
        f"Above Upper Bound: {actual_above:.1%} (expected: {expected_above:.1%}) {status}"
    )

    # Interval width statistics
    print(f"\nInterval Width Statistics:")
    print(f"  Average: {coverage_results['avg_interval_width']:.4f}")
    print(f"  Median:  {coverage_results['median_interval_width']:.4f}")
    print(f"  Min:     {coverage_results['min_interval_width']:.4f}")
    print(f"  Max:     {coverage_results['max_interval_width']:.4f}")

    # Recommendations
    if not coverage_ok:
        print("\n⚠️  CALIBRATION RECOMMENDATIONS:")
        if actual_coverage < target_coverage - coverage_tolerance:
            print(f"   Coverage too low ({actual_coverage:.1%}). Consider:")
            print(f"   - Widening quantiles (e.g., α=0.01/0.99 instead of 0.025/0.975)")
            print(f"   - Applying post-hoc calibration adjustment")
        else:
            print(f"   Coverage too high ({actual_coverage:.1%}). Consider:")
            print(
                f"   - Narrowing quantiles (e.g., α=0.05/0.95 instead of 0.025/0.975)"
            )

    return coverage_ok


def print_example_predictions(y_true, y_pred, y_lower, y_upper, n_examples=10):
    """Print example predictions with intervals."""
    print("\n" + "=" * 60)
    print("EXAMPLE PREDICTIONS WITH 95% INTERVALS (Price Space)")
    print("=" * 60)
    print(
        f"{'Actual':>12} {'Predicted':>12} {'Lower':>12} {'Upper':>12} {'In Interval':>12}"
    )
    print("-" * 60)

    # Select random examples
    indices = np.random.choice(len(y_true), min(n_examples, len(y_true)), replace=False)

    for idx in indices:
        actual = y_true.iloc[idx] if hasattr(y_true, "iloc") else y_true[idx]
        pred = y_pred[idx]
        lower = y_lower[idx]
        upper = y_upper[idx]
        in_interval = "✓" if lower <= actual <= upper else "✗"

        print(
            f"${actual:>11,.2f} ${pred:>11,.2f} ${lower:>11,.2f} ${upper:>11,.2f} {in_interval:>12}"
        )


def main():
    data_dir = "model/data"
    results_dir = "model/results"

    best_params = find_best_grid(results_dir)
    if not best_params:
        print("Could not find suitable parameters. Exiting.")
        return

    X_train, y_train, X_val, y_val, X_test, y_test = load_data(data_dir)

    print("Combining Train and Validation sets for final training...")
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])

    print(f"Final training set size: {X_train_full.shape[0]}")

    # Get device
    device = get_xgb_device()

    # Train point prediction model (existing approach)
    print("\nTraining point prediction model...")
    model = XGBRegressor(device=device, **best_params)
    model.fit(X_train_full, y_train_full)

    # Train quantile models for prediction intervals
    model_lower, model_upper = train_quantile_models(
        X_train_full, y_train_full, best_params, device
    )

    # Generate predictions (log space)
    print("\nGenerating predictions on Test set...")
    y_test_pred_log = model.predict(X_test)
    y_test_lower_log = model_lower.predict(X_test)
    y_test_upper_log = model_upper.predict(X_test)

    # Evaluate coverage in log space
    coverage_log = evaluate_coverage(
        y_test.values, y_test_lower_log, y_test_upper_log, space_name="log"
    )

    # Transform to price space
    y_test_price = np.exp(y_test)
    y_test_pred_price = np.exp(y_test_pred_log)
    y_test_lower_price = np.exp(y_test_lower_log)
    y_test_upper_price = np.exp(y_test_upper_log)

    # Evaluate coverage in price space
    coverage_price = evaluate_coverage(
        y_test_price.values, y_test_lower_price, y_test_upper_price, space_name="price"
    )

    # Standard metrics (existing)
    val_rmse = np.sqrt(mean_squared_error(y_test_price, y_test_pred_price))
    val_mae = mean_absolute_error(y_test_price, y_test_pred_price)
    val_mape = mean_absolute_percentage_error(y_test_price, y_test_pred_price)
    val_r2 = r2_score(y_test_price, y_test_pred_price)

    output_lines = []

    try:
        if "prev_1_price" in X_test.columns:
            simple_percent_error = (
                np.abs(X_test["prev_1_price"].values - y_test.values) / y_test.values
            ) * 100
            simple_percent_error_series = pd.Series(
                simple_percent_error, name="simple_percent_error"
            )
            output_lines.append("\nSimple Percent Error Percentiles (Baseline):")
            output_lines.append(
                simple_percent_error_series.describe(
                    percentiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                ).to_string()
            )
    except Exception as e:
        print(f"Could not calc simple percent error validation: {e}")

    percent_error = (
        np.abs(y_test_pred_price - y_test_price.values) / y_test_price.values
    ) * 100
    percent_error_series = pd.Series(percent_error, name="percent_error")

    output_lines.append("\nTest Metrics:")
    output_lines.append(f"  RMSE: ${val_rmse:,.2f}")
    output_lines.append(f"  MAE:  ${val_mae:,.2f}")
    output_lines.append(f"  MAPE: {val_mape:.2%}")
    output_lines.append(f"  R²:   {val_r2:.4f}")

    output_lines.append("\nModel Percent Error Percentiles:")
    output_lines.append(
        percent_error_series.describe(
            percentiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        ).to_string()
    )

    # Add coverage metrics to output
    output_lines.append("\n" + "=" * 50)
    output_lines.append("PREDICTION INTERVAL METRICS")
    output_lines.append("=" * 50)
    output_lines.append(f"\nQuantile Configuration:")
    output_lines.append(f"  Lower quantile (α): {LOWER_QUANTILE}")
    output_lines.append(f"  Upper quantile (α): {UPPER_QUANTILE}")
    output_lines.append(f"  Target coverage: {TARGET_COVERAGE:.0%}")

    output_lines.append(f"\nLog Space Coverage:")
    output_lines.append(f"  Coverage: {coverage_log['coverage']:.1%}")
    output_lines.append(f"  Below lower bound: {coverage_log['below_lower']:.1%}")
    output_lines.append(f"  Above upper bound: {coverage_log['above_upper']:.1%}")
    output_lines.append(
        f"  Avg interval width: {coverage_log['avg_interval_width']:.4f}"
    )

    output_lines.append(f"\nPrice Space Coverage:")
    output_lines.append(f"  Coverage: {coverage_price['coverage']:.1%}")
    output_lines.append(f"  Below lower bound: {coverage_price['below_lower']:.1%}")
    output_lines.append(f"  Above upper bound: {coverage_price['above_upper']:.1%}")
    output_lines.append(
        f"  Avg interval width: ${coverage_price['avg_interval_width']:,.2f}"
    )
    output_lines.append(
        f"  Median interval width: ${coverage_price['median_interval_width']:,.2f}"
    )

    # Feature Importance
    importance_df = pd.DataFrame(
        {"feature": X_train_full.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    output_lines.append("\nTop 20 Feature Importances:")
    output_lines.append(importance_df.head(20).to_string())

    # Print and Save
    full_output = "\n".join(output_lines)
    print(full_output)

    # Print detailed coverage reports
    print_coverage_report(coverage_log)
    print_coverage_report(coverage_price)

    # Print example predictions
    print_example_predictions(
        y_test_price, y_test_pred_price, y_test_lower_price, y_test_upper_price
    )

    metrics_path = os.path.join(results_dir, "test_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(full_output)
    print(f"\nSaved test metrics to {metrics_path}")

    # Save coverage results as JSON for programmatic access
    coverage_results = {
        "quantile_config": {
            "lower_alpha": LOWER_QUANTILE,
            "upper_alpha": UPPER_QUANTILE,
            "target_coverage": TARGET_COVERAGE,
        },
        "log_space": coverage_log,
        "price_space": coverage_price,
    }
    coverage_path = os.path.join(results_dir, "coverage_metrics.json")
    with open(coverage_path, "w") as f:
        json.dump(coverage_results, f, indent=2)
    print(f"Saved coverage metrics to {coverage_path}")

    # Feature importance plot
    plt.figure(figsize=(10, 12))
    plt.barh(importance_df["feature"][:20], importance_df["importance"][:20])
    plt.xlabel("Importance")
    plt.title("Top 20 Feature Importances")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "feature_importance.png")
    plt.savefig(plot_path)
    print(f"Saved feature importance plot to {plot_path}")

    # Save trained models for inference API
    artifacts_dir = "api/artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)

    print("\nSaving trained models for inference...")
    model.save_model(os.path.join(artifacts_dir, "model_point.json"))
    model_lower.save_model(os.path.join(artifacts_dir, "model_lower.json"))
    model_upper.save_model(os.path.join(artifacts_dir, "model_upper.json"))
    print(f"  Saved model_point.json")
    print(f"  Saved model_lower.json")
    print(f"  Saved model_upper.json")

    # Save model config for API
    model_config = {
        "lower_quantile": LOWER_QUANTILE,
        "upper_quantile": UPPER_QUANTILE,
        "target_coverage": TARGET_COVERAGE,
        "best_params": best_params,
    }
    config_path = os.path.join(artifacts_dir, "model_config.json")
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2)
    print(f"  Saved model_config.json")

    # Copy feature columns to artifacts
    import shutil

    feature_cols_src = os.path.join(data_dir, "feature_cols.json")
    feature_cols_dst = os.path.join(artifacts_dir, "feature_cols.json")
    if os.path.exists(feature_cols_src):
        shutil.copy(feature_cols_src, feature_cols_dst)
        print(f"  Copied feature_cols.json")

    print(f"\nAll artifacts saved to {artifacts_dir}/")


if __name__ == "__main__":
    main()
