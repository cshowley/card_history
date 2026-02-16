import glob
import json
import os

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

    global_best_score = float("inf")
    global_best_params = {}

    if not result_files:
        print("No result files found in model/results/.")
        return None

    for f in result_files:
        try:
            with open(f, "r") as fp:
                data = json.load(fp)
                score = data.get("best_score", float("inf"))
                metric_name = data.get("metric_name", "unknown")
                print(f"File {f}: {metric_name} = {score:.6f}")
                if score < global_best_score:
                    global_best_score = score
                    global_best_params = data.get("best_params", {})
        except Exception as e:
            print(f"Error reading {f}: {e}")

    print(f"Global Best Score: {global_best_score:.6f}")
    print(f"Global Best Params: {global_best_params}")
    return global_best_params


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

    print("Training final model...")
    model = XGBRegressor(device="cuda", **best_params)
    model.fit(X_train_full, y_train_full)

    print("Predicting on Test set...")
    y_test_pred = model.predict(X_test)
    val_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    val_mae = mean_absolute_error(y_test, y_test_pred)
    val_mape = mean_absolute_percentage_error(y_test, y_test_pred)
    val_r2 = r2_score(y_test, y_test_pred)

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

    percent_error = (np.abs(y_test_pred - y_test.values) / y_test.values) * 100
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

    # Feature Importance
    importance_df = pd.DataFrame(
        {"feature": X_train_full.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    output_lines.append("\nTop 20 Feature Importances:")
    output_lines.append(importance_df.head(20).to_string())

    # Print and Save
    full_output = "\n".join(output_lines)
    print(full_output)

    metrics_path = os.path.join(results_dir, "test_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(full_output)
    print(f"\nSaved test metrics to {metrics_path}")

    plt.figure(figsize=(10, 12))
    plt.barh(importance_df["feature"][:20], importance_df["importance"][:20])
    plt.xlabel("Importance")
    plt.title("Top 20 Feature Importances")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "feature_importance.png")
    plt.savefig(plot_path)
    print(f"Saved feature importance plot to {plot_path}")


if __name__ == "__main__":
    main()
