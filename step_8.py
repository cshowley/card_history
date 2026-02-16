"""
Step 8: Run inference on today's data with bin-based model routing.

This module loads the primary (bin-specific) models and the fallback model,
routing cards to the appropriate model:
- Cards WITH price history: Routed to the appropriate bin model based on last_known_price
- Cards WITHOUT price history: Routed to the fallback model

Output columns:
- prediction: The predicted price (single consolidated column)
- prediction_model: Which model made the prediction ("bin_0", "bin_1", ..., or "fallback")
"""

import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb
from tqdm import tqdm

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
        Array of bin indices (0 to n_bins-1). NaN prices get index -1.
    """
    # Handle NaN values by setting them to -1 (will use fallback model)
    result = np.full(len(prices), -1, dtype=int)
    valid_mask = ~np.isnan(prices)

    if valid_mask.any():
        valid_prices = prices[valid_mask]
        bin_indices = np.searchsorted(bin_edges, valid_prices, side="right") - 1
        result[valid_mask] = np.clip(bin_indices, 0, len(bin_edges) - 2)

    return result


def get_bin_model_path(bin_idx: int) -> str:
    """Get the output file path for a bin-specific model."""
    return os.path.join(
        constants.S7_PRIMARY_MODEL_OUTPUT_DIR, f"xgb_model_bin_{bin_idx}.json"
    )


def get_fallback_model_path() -> str:
    """Get the output file path for the fallback model."""
    return os.path.join(constants.S7_FALLBACK_MODEL_OUTPUT_DIR, "xgb_model_all.json")


def load_bin_models(device: str) -> dict:
    """
    Load all bin-specific models for the primary configuration.

    Args:
        device: Device string for XGBoost.

    Returns:
        Dictionary mapping bin_idx to loaded XGBoost model.
    """
    n_bins = len(constants.S7_CUSTOM_BIN_LABELS)
    bin_models = {}

    for bin_idx in range(n_bins):
        model_file = get_bin_model_path(bin_idx)
        if os.path.exists(model_file):
            model = xgb.XGBRegressor()
            model.load_model(model_file)
            model.set_params(device=device)
            bin_models[bin_idx] = model

    return bin_models


def load_fallback_model(device: str) -> xgb.XGBRegressor | None:
    """
    Load the fallback model.

    Args:
        device: Device string for XGBoost.

    Returns:
        Loaded XGBoost model, or None if not found.
    """
    model_file = get_fallback_model_path()
    if os.path.exists(model_file):
        model = xgb.XGBRegressor()
        model.load_model(model_file)
        model.set_params(device=device)
        return model
    return None


def run_step_8():
    """Run inference on today's data using bin-specific and fallback models."""
    print("Starting Step 8: Inference on Today's Data with Bin Routing...")

    # Get model configurations from constants
    primary_dataset = constants.S7_PRIMARY_MODEL_DATASET
    fallback_dataset = constants.S7_FALLBACK_MODEL_DATASET
    primary_objective = constants.S7_PRIMARY_MODEL_OBJECTIVE
    fallback_objective = constants.S7_FALLBACK_MODEL_OBJECTIVE
    primary_is_log = primary_dataset == "log"

    output_file = constants.S8_PREDICTIONS_FILE
    batch_size = constants.S6_BATCH_SIZE
    exclude_cols = constants.S7_EXCLUDE_COLS

    print(f"Primary model: dataset={primary_dataset}, objective={primary_objective}")
    print(f"Fallback model: dataset={fallback_dataset}, objective={fallback_objective}")
    print(f"Custom bin labels: {constants.S7_CUSTOM_BIN_LABELS}")

    # Load models
    print("\nLoading models...")
    bin_models = load_bin_models(device="cuda:0")
    fallback_model = load_fallback_model(device="cuda:1")

    print(f"  Loaded {len(bin_models)} bin models for primary")
    print(f"  Fallback model loaded: {fallback_model is not None}")

    if not bin_models and fallback_model is None:
        print("No models could be loaded. Exiting.")
        return

    # Get dataset files
    primary_dataset_files = constants.get_dataset_files(primary_dataset)
    fallback_dataset_files = constants.get_dataset_files(fallback_dataset)

    primary_input_file = primary_dataset_files["today_file"].replace(
        ".parquet", "_with_neighbors.parquet"
    )
    fallback_input_file = fallback_dataset_files["today_file"].replace(
        ".parquet", "_with_neighbors.parquet"
    )

    if not os.path.exists(primary_input_file):
        raise FileNotFoundError(f"Primary input file not found: {primary_input_file}")

    print(f"\nPrimary dataset file: {primary_input_file}")
    print(f"Fallback dataset file: {fallback_input_file}")

    # Open parquet files
    primary_parquet = pq.ParquetFile(primary_input_file)

    # If datasets are different, open fallback parquet file separately
    fallback_parquet = None
    fallback_iterator = None
    if fallback_dataset != primary_dataset:
        if os.path.exists(fallback_input_file):
            fallback_parquet = pq.ParquetFile(fallback_input_file)
            fallback_iterator = fallback_parquet.iter_batches(batch_size=batch_size)
            print(f"Using separate fallback dataset: {fallback_input_file}")
        else:
            print(f"Warning: Fallback dataset file not found: {fallback_input_file}")
            print("Will use primary dataset features for fallback predictions")

    # Get bin edges for primary model
    bin_edges = get_bin_edges(is_log_prices=primary_is_log)

    writer = None
    total_rows = 0
    stats = {"bin_counts": {}, "fallback_count": 0}

    for batch in tqdm(
        primary_parquet.iter_batches(batch_size=batch_size), desc="Running Inference"
    ):
        df_primary = batch.to_pandas()

        # Load corresponding fallback batch if using separate datasets
        df_fallback = None
        if fallback_iterator is not None:
            try:
                fallback_batch = next(fallback_iterator)
                df_fallback = fallback_batch.to_pandas()
            except StopIteration:
                print("Warning: Fallback dataset exhausted before primary dataset")
                df_fallback = None

        # Build identity dataframe with key columns
        ident_df = df_primary[["gemrate_id", "grade", "half_grade"]].copy()

        # Extract grading company from one-hot encoded columns
        conditions_gc = [
            df_primary.get("grade_co_PSA", pd.Series([0] * len(df_primary))) == 1,
            df_primary.get("grade_co_BGS", pd.Series([0] * len(df_primary))) == 1,
            df_primary.get("grade_co_CGC", pd.Series([0] * len(df_primary))) == 1,
        ]
        choices_gc = ["PSA", "BGS", "CGC"]
        ident_df["grading_company"] = np.select(
            conditions_gc, choices_gc, default="Unknown"
        )

        # Get feature columns
        feature_cols_primary = [c for c in df_primary.columns if c not in exclude_cols]
        X_primary = df_primary[feature_cols_primary]

        # Get features for fallback (may be from different dataset)
        if df_fallback is not None:
            feature_cols_fallback = [
                c for c in df_fallback.columns if c not in exclude_cols
            ]
            X_fallback = df_fallback[feature_cols_fallback]
        else:
            # Use primary features for fallback if no separate dataset
            X_fallback = X_primary
            feature_cols_fallback = feature_cols_primary

        # Get last_known_price for bin routing
        # Note: last_known_price scale matches the primary dataset type
        if "last_known_price" in df_primary.columns:
            last_known_price = df_primary["last_known_price"].values.copy()
        else:
            # No last_known_price column - all go to fallback
            last_known_price = np.full(len(df_primary), np.nan)

        # Assign samples to bins
        bin_indices = assign_to_bins(last_known_price, bin_edges)

        # Initialize predictions array and model tracking
        predictions = np.zeros(len(df_primary))
        prediction_model = np.empty(len(df_primary), dtype=object)

        # Process each bin (samples with price history)
        for bin_idx in range(len(constants.S7_CUSTOM_BIN_LABELS)):
            bin_mask = bin_indices == bin_idx

            if not bin_mask.any():
                continue

            if bin_idx in bin_models:
                model = bin_models[bin_idx]
                X_bin = X_primary[bin_mask]
                preds = model.predict(X_bin)
                predictions[bin_mask] = preds
                prediction_model[bin_mask] = f"bin_{bin_idx}"

                # Track stats
                count = bin_mask.sum()
                stats["bin_counts"][bin_idx] = (
                    stats["bin_counts"].get(bin_idx, 0) + count
                )
            else:
                # No bin model available - will be handled by fallback
                bin_indices[bin_mask] = -1

        # Process fallback (samples without price history OR missing bin models)
        fallback_mask = bin_indices == -1
        if fallback_mask.any():
            if fallback_model is not None:
                X_fb = X_fallback[fallback_mask]
                preds = fallback_model.predict(X_fb)
                predictions[fallback_mask] = preds
                prediction_model[fallback_mask] = "fallback"
                stats["fallback_count"] += fallback_mask.sum()
            else:
                # No fallback model - set to NaN
                predictions[fallback_mask] = np.nan
                prediction_model[fallback_mask] = "none"

        # Add predictions to identity dataframe
        ident_df["prediction"] = predictions
        ident_df["prediction_model"] = prediction_model

        table = pa.Table.from_pandas(ident_df)

        if writer is None:
            writer = pq.ParquetWriter(output_file, table.schema)

        writer.write_table(table)
        total_rows += len(ident_df)

    if writer:
        writer.close()
        print(f"\nInference complete. Saved {total_rows} predictions to {output_file}.")

        # Print routing statistics
        print("\nRouting Statistics:")
        print("\n  PRIMARY MODEL (bin-specific):")
        for bin_idx, count in sorted(stats["bin_counts"].items()):
            label = constants.S7_CUSTOM_BIN_LABELS[bin_idx]
            print(f"    Bin {bin_idx} ({label}): {count:,} samples")
        print("\n  FALLBACK MODEL:")
        print(f"    No price history: {stats['fallback_count']:,} samples")
    else:
        print("No data processed.")


if __name__ == "__main__":
    run_step_8()
