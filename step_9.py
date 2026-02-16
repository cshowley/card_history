"""
Step 9: Re-sort predictions to enforce monotonicity.

This module processes predictions from Step 8, ensuring that predictions
are monotonically increasing within each (gemrate_id, grading_company) group
when sorted by grade.

It also handles exponentiation of predictions from log-transformed models:
- If the primary model uses log dataset, bin predictions are exponentiated
- If the fallback model uses log dataset, fallback predictions are exponentiated
"""

import os

import numpy as np
import pandas as pd

import constants


def run_step_9():
    """Re-sort predictions to enforce monotonicity across grades."""
    print("Starting Step 9: Re-sorting Predictions...")

    input_file = constants.S8_PREDICTIONS_FILE
    output_file = constants.S9_OUTPUT_FILE

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Get model dataset types from constants
    primary_dataset = constants.S7_PRIMARY_MODEL_DATASET
    fallback_dataset = constants.S7_FALLBACK_MODEL_DATASET
    primary_is_log = primary_dataset == "log"
    fallback_is_log = fallback_dataset == "log"

    print(f"Primary model dataset: {primary_dataset} (log={primary_is_log})")
    print(f"Fallback model dataset: {fallback_dataset} (log={fallback_is_log})")

    print(f"Reading {input_file}...")
    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df)} rows.")

    # Verify required columns exist
    if "prediction" not in df.columns:
        raise ValueError("Required column 'prediction' not found in data.")
    if "prediction_model" not in df.columns:
        raise ValueError("Required column 'prediction_model' not found in data.")

    # Exponentiate predictions based on which model made them
    # Only exponentiate if the model that made the prediction used log dataset
    if primary_is_log:
        # Bin models use log dataset - exponentiate their predictions
        bin_mask = df["prediction_model"].str.startswith("bin_")
        bin_count = bin_mask.sum()
        print(
            f"Exponentiating {bin_count:,} predictions from bin models (log-transformed)..."
        )
        df.loc[bin_mask, "prediction"] = np.exp(df.loc[bin_mask, "prediction"])

    if fallback_is_log:
        # Fallback model uses log dataset - exponentiate its predictions
        fallback_mask = df["prediction_model"] == "fallback"
        fallback_count = fallback_mask.sum()
        print(
            f"Exponentiating {fallback_count:,} predictions from fallback model (log-transformed)..."
        )
        df.loc[fallback_mask, "prediction"] = np.exp(
            df.loc[fallback_mask, "prediction"]
        )

    # Sort dataframe by group columns and grade
    sort_cols = ["gemrate_id", "grading_company", "grade", "half_grade"]
    print(f"Sorting dataframe by {sort_cols}...")
    df = df.sort_values(by=sort_cols, ascending=True).reset_index(drop=True)

    # Group by gemrate_id and grading_company
    group_cols = ["gemrate_id", "grading_company"]
    total_groups = df.groupby(group_cols).ngroups

    # Enforce monotonicity: sort predictions within each group
    print("Processing predictions column for monotonicity...")

    # Count groups that violate monotonicity before sorting
    violation_counts = (
        df.groupby(group_cols)["prediction"]
        .apply(lambda x: np.any(x.values[:-1] > x.values[1:]))
        .sum()
    )
    print(f"  Found {violation_counts} / {total_groups} groups violating monotonicity.")

    # Sort values within each group to enforce monotonicity
    df["prediction"] = df.groupby(group_cols)["prediction"].transform(
        lambda x: np.sort(x.values)
    )

    print(f"Saving sorted predictions to {output_file}...")
    df.to_parquet(output_file)

    # Print summary statistics
    print(f"\nDone. Saved {len(df)} rows.")
    print(f"Output columns: {list(df.columns)}")

    # Print prediction model distribution
    print("\nPrediction model distribution:")
    model_counts = df["prediction_model"].value_counts()
    for model_name, count in model_counts.items():
        print(f"  {model_name}: {count:,} ({100 * count / len(df):.1f}%)")


if __name__ == "__main__":
    run_step_9()
