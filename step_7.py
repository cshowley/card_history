"""
Step 7: Train XGBoost models with custom bin strategy.

This module trains two types of models:
1. Primary model: Bin-specific models for samples with price history
   - One model per price bin (e.g., $0-$32, $32-$64, etc.)
   - Uses objective and dataset from constants.S7_PRIMARY_MODEL_*
2. Fallback model: Single model for samples without price history
   - Trained on all data
   - Uses objective and dataset from constants.S7_FALLBACK_MODEL_*

Best hyperparameters are loaded from the directories specified in constants.
"""

import glob
import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb

import constants


def get_bin_edges(is_log_prices: bool = False) -> np.ndarray:
    """
    Get bin edges, optionally converted to log scale.

    Args:
        is_log_prices: If True, convert dollar edges to log scale.

    Returns:
        Array of bin edges in the appropriate scale.
    """
    edges = np.array(constants.S7_CUSTOM_BIN_EDGES, dtype=float)

    if is_log_prices:
        # Handle inf specially - use a very large number before log
        edges = np.where(edges >= float("inf"), 1e12, edges)
        edges = np.where(edges <= 0, 0.01, edges)  # Avoid log(0)
        return np.log(edges)
    else:
        # For normal prices, use large number for inf
        return np.where(edges >= float("inf"), 1e12, edges)


def assign_to_bins(prices: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """
    Assign an array of prices to bin indices.

    Args:
        prices: Array of prices.
        bin_edges: Array of bin edges.

    Returns:
        Array of bin indices (0 to n_bins-1).
    """
    bin_indices = np.searchsorted(bin_edges, prices, side="right") - 1
    return np.clip(bin_indices, 0, len(bin_edges) - 2)


def get_best_params(results_dir: str) -> dict:
    """
    Finds the best hyperparameters from the specified results directory.

    Args:
        results_dir: Path to the results directory.

    Returns:
        Dictionary of best hyperparameters.

    Raises:
        FileNotFoundError: If no result files found.
        ValueError: If no valid parameters found.
    """
    result_files = glob.glob(os.path.join(results_dir, "worker_*_best.json"))

    if not result_files:
        raise FileNotFoundError(f"No result files found in {results_dir}")

    metric_key = "best_score"

    best_metric = float("inf")
    best_params = None
    best_worker = None

    for file in result_files:
        with open(file, "r") as f:
            data = json.load(f)
            if metric_key in data and "best_params" in data:
                if data[metric_key] < best_metric:
                    best_metric = data[metric_key]
                    best_params = data["best_params"]
                    best_worker = data.get("worker_id", "unknown")

    if best_params is None:
        raise ValueError(f"Could not find any valid best parameters in {results_dir}")

    print(
        f"Best params found from worker {best_worker} with {metric_key}: {best_metric:.6f}"
    )
    return best_params


def load_dataset(dataset_type: str):
    """
    Load the dataset for the specified dataset type.

    Args:
        dataset_type: Either "normal" or "log"

    Returns:
        Tuple of (X, y, df) where X is the feature DataFrame, y is the target Series,
        and df is the full DataFrame.
    """
    dataset_files = constants.get_dataset_files(dataset_type)
    input_file = dataset_files["historical_file"].replace(
        ".parquet", "_with_neighbors.parquet"
    )

    print(f"Loading data from {input_file}...")
    df = pd.read_parquet(input_file)
    df = df.dropna(subset=["price"])

    df["grade"] = pd.to_numeric(df["grade"], errors="raise")
    df["half_grade"] = pd.to_numeric(df["half_grade"], errors="raise")
    df["seller_popularity"] = pd.to_numeric(df["seller_popularity"], errors="raise")

    exclude_cols = constants.S7_EXCLUDE_COLS
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols]
    y = df["price"]

    return X, y, df


def get_bin_model_path(bin_idx: int) -> str:
    """Get the output file path for a bin-specific model."""
    return os.path.join(
        constants.S7_PRIMARY_MODEL_OUTPUT_DIR, f"xgb_model_bin_{bin_idx}.json"
    )


def get_fallback_model_path() -> str:
    """Get the output file path for the fallback model."""
    return os.path.join(constants.S7_FALLBACK_MODEL_OUTPUT_DIR, "xgb_model_all.json")


def train_model():
    """Trains the XGBoost models with bin-specific strategy."""
    print("Step 7: Training XGBoost Models with Custom Bin Strategy")

    print(f"Custom bin edges: {constants.S7_CUSTOM_BIN_EDGES}")
    print(f"Number of bins: {len(constants.S7_CUSTOM_BIN_LABELS)}")

    # =========================================================================
    # Train Primary Model (bin-specific models)
    # =========================================================================
    primary_objective = constants.S7_PRIMARY_MODEL_OBJECTIVE
    primary_dataset = constants.S7_PRIMARY_MODEL_DATASET
    primary_output_dir = constants.S7_PRIMARY_MODEL_OUTPUT_DIR
    primary_results_dir = constants.S7_PRIMARY_MODEL_RESULTS_DIR
    is_log = primary_dataset == "log"

    # Create output directory for primary model
    os.makedirs(primary_output_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("[PRIMARY] Training bin-specific models")
    print(f"Dataset: {primary_dataset}")
    print(f"Objective: {primary_objective}")
    print(f"Output directory: {primary_output_dir}")
    print(f"Results directory: {primary_results_dir}")
    print(f"{'=' * 60}")

    # Load dataset for primary model
    X_primary, y_primary, df_primary = load_dataset(primary_dataset)
    print(f"[PRIMARY] Loaded {len(X_primary)} samples")

    # Get best hyperparameters for primary model
    print("[PRIMARY] Getting best hyperparameters...")
    try:
        primary_params = get_best_params(primary_results_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"[PRIMARY] Warning - {e}")
        print("Using default hyperparameters")
        primary_params = {}

    # Prepare model parameters
    worker_params_primary = {
        k: v for k, v in primary_params.items() if k != "objective"
    }
    worker_params_primary["device"] = "cuda:0"

    # Get bin edges in the appropriate scale
    bin_edges = get_bin_edges(is_log_prices=is_log)
    bin_labels = constants.S7_CUSTOM_BIN_LABELS

    # Assign training samples to bins
    train_prices = np.array(y_primary.values)
    train_bin_indices = assign_to_bins(train_prices, bin_edges)

    print(f"\n[PRIMARY] Training {len(bin_labels)} bin-specific models...")

    # Train a model for each bin
    for bin_idx, bin_label in enumerate(bin_labels):
        bin_mask = train_bin_indices == bin_idx
        bin_count = bin_mask.sum()

        if bin_count < 10:
            print(f"  Bin {bin_idx} ({bin_label}): Skipping (only {bin_count} samples)")
            continue

        X_bin = X_primary[bin_mask]
        y_bin = y_primary[bin_mask]

        print(f"  Bin {bin_idx} ({bin_label}): Training on {bin_count:,} samples...")

        model = xgb.XGBRegressor(objective=primary_objective, **worker_params_primary)
        model.fit(X_bin, y_bin)

        output_file = get_bin_model_path(bin_idx)
        model.save_model(output_file)
        print(f"    Saved to {output_file}")

    print("\n[PRIMARY] Finished successfully.")

    # =========================================================================
    # Train Fallback Model (single model on all data)
    # =========================================================================
    fallback_objective = constants.S7_FALLBACK_MODEL_OBJECTIVE
    fallback_dataset = constants.S7_FALLBACK_MODEL_DATASET
    fallback_output_dir = constants.S7_FALLBACK_MODEL_OUTPUT_DIR
    fallback_results_dir = constants.S7_FALLBACK_MODEL_RESULTS_DIR

    # Create output directory for fallback model
    os.makedirs(fallback_output_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("[FALLBACK] Training fallback model")
    print(f"Dataset: {fallback_dataset}")
    print(f"Objective: {fallback_objective}")
    print(f"Output directory: {fallback_output_dir}")
    print(f"Results directory: {fallback_results_dir}")
    print(f"{'=' * 60}")

    # Load dataset for fallback model (may be different from primary)
    if fallback_dataset == primary_dataset:
        X_fallback, y_fallback = X_primary, y_primary
        print(f"[FALLBACK] Using same dataset as primary ({len(X_fallback)} samples)")
    else:
        X_fallback, y_fallback, _ = load_dataset(fallback_dataset)
        print(f"[FALLBACK] Loaded {len(X_fallback)} samples")

    # Get best hyperparameters for fallback model
    print("[FALLBACK] Getting best hyperparameters...")
    try:
        fallback_params = get_best_params(fallback_results_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"[FALLBACK] Warning - {e}")
        print("Using default hyperparameters")
        fallback_params = {}

    # Prepare model parameters
    worker_params_fallback = {
        k: v for k, v in fallback_params.items() if k != "objective"
    }
    worker_params_fallback["device"] = "cuda:0"

    # Train the fallback model on all data
    print(f"\n[FALLBACK] Training on all {len(X_fallback):,} samples...")
    fallback_model = xgb.XGBRegressor(
        objective=fallback_objective, **worker_params_fallback
    )
    fallback_model.fit(X_fallback, y_fallback)

    fallback_model_file = get_fallback_model_path()
    fallback_model.save_model(fallback_model_file)
    print(f"  Saved to {fallback_model_file}")

    print("\n[FALLBACK] Finished successfully.")

    print("\n" + "=" * 60)
    print("All models trained and saved successfully.")
    print("=" * 60)


def run_step_7():
    try:
        train_model()
    except Exception as e:
        print(f"Failed to run Step 7: {e}")
        raise e


if __name__ == "__main__":
    run_step_7()
