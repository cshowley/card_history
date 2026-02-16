"""
Model setup and hyperparameter search orchestration.

This module handles data preparation and launches hyperparameter search workers
for each enabled model defined in model_config.json.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from config_loader import (
    load_model_config,
    get_enabled_models,
    get_training_config,
    get_data_config,
    get_dataset_files,
    get_random_params,
    get_results_dir,
)

START_DATE = datetime(2025, 9, 8) + timedelta(days=28)


def load_and_prep_data(df, feature_cols, config, dataset_type="normal"):
    """
    Split data into train/val/test sets and save to disk.

    Args:
        df: Input DataFrame with features and target.
        feature_cols: List of feature column names.
        config: Model configuration dictionary.
        dataset_type: The dataset type ("normal" or "log").
    """
    training_config = get_training_config(config)
    train_test_split = training_config["train_test_split"]
    val_test_split = training_config["val_test_split"]

    train_df = df.iloc[: int(len(df) * train_test_split)]
    test_df = df.iloc[int(len(df) * train_test_split) :]
    val_df = test_df.iloc[: int(len(test_df) * val_test_split)]
    test_df = test_df.iloc[int(len(test_df) * val_test_split) :]

    X_train = train_df[feature_cols]
    y_train = train_df["price"]

    X_val = val_df[feature_cols]
    y_val = val_df["price"]

    X_test = test_df[feature_cols]
    y_test = test_df["price"]

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    data_dir = f"model/data/{dataset_type}"
    os.makedirs(data_dir, exist_ok=True)

    print(f"Saving datasets to {data_dir}...")
    X_train.to_parquet(f"{data_dir}/X_train.parquet")
    y_train.to_pickle(f"{data_dir}/y_train.pkl")
    X_val.to_parquet(f"{data_dir}/X_val.parquet")
    y_val.to_pickle(f"{data_dir}/y_val.pkl")
    X_test.to_parquet(f"{data_dir}/X_test.parquet")
    y_test.to_pickle(f"{data_dir}/y_test.pkl")

    with open(f"{data_dir}/feature_cols.json", "w") as f:
        json.dump(feature_cols, f)


def run_hyperparameter_search_for_model(model_config, config):
    """
    Run hyperparameter search for a single model.

    Args:
        model_config: Configuration for the specific model.
        config: Full model configuration dictionary.
    """
    model_name = model_config["name"]
    dataset_type = model_config.get("dataset", "normal")
    training_config = get_training_config(config)
    n_workers = training_config["n_workers"]
    n_trials = training_config["n_trials"]

    print(f"\n{'=' * 60}")
    print(f"Starting hyperparameter search for model: {model_name}")
    print(f"Dataset: {dataset_type}")
    print(f"Objective: {model_config['objective']}")
    print(f"{'=' * 60}")

    print(f"Generating {n_trials} random parameter combinations...")
    random_search_grid = [get_random_params(config) for _ in range(n_trials)]

    chunk_size = int(np.ceil(n_trials / n_workers))
    chunks = [
        random_search_grid[i : i + chunk_size] for i in range(0, n_trials, chunk_size)
    ]

    config_dir = "model/configs"
    os.makedirs(config_dir, exist_ok=True)

    # Create results directory for this model
    results_dir = get_results_dir(model_name)
    os.makedirs(results_dir, exist_ok=True)

    processes = []

    print(f"Launching {n_workers} workers for {model_name}...")
    for i, chunk in enumerate(chunks):
        if not chunk:
            continue

        config_path = f"{config_dir}/grid_chunk_{model_name}_{i}.json"
        with open(config_path, "w") as f:
            json.dump(chunk, f)

        cmd = [
            sys.executable,
            "model/hyperparameter_search_worker.py",
            "--gpu_id",
            str(i),
            "--config",
            config_path,
            "--model_name",
            model_name,
            "--dataset",
            dataset_type,
        ]

        p = subprocess.Popen(cmd)
        processes.append(p)
        print(f"Started worker {i} for {model_name} with {len(chunk)} combinations.")

    for p in processes:
        p.wait()

    print(f"All workers finished for model: {model_name}")


def split_grid_and_run_workers(config):
    """
    Run hyperparameter search sequentially for all enabled models.

    Args:
        config: Full model configuration dictionary.
    """
    enabled_models = get_enabled_models(config)

    if not enabled_models:
        print("No models enabled in config. Nothing to do.")
        return

    print(
        f"Found {len(enabled_models)} enabled models: {[m['name'] for m in enabled_models]}"
    )

    for model_config in enabled_models:
        run_hyperparameter_search_for_model(model_config, config)

    print("\n" + "=" * 60)
    print("All hyperparameter searches completed.")
    print("=" * 60)


if __name__ == "__main__":
    print("Loading configuration...")
    config = load_model_config()
    data_config = get_data_config(config)
    enabled_models = get_enabled_models(config)

    if not enabled_models:
        print("No models enabled in config. Nothing to do.")
        exit(0)

    # Find unique dataset types used by enabled models
    dataset_types = set(m.get("dataset", "normal") for m in enabled_models)
    print(f"Dataset types needed: {dataset_types}")

    bad_features = data_config["bad_features"]
    exclude_cols = data_config["exclude_cols"]

    # Prepare data for each dataset type
    for dataset_type in dataset_types:
        print(f"\n{'=' * 60}")
        print(f"Preparing data for dataset: {dataset_type}")
        print(f"{'=' * 60}")

        # Get dataset files for this type (use any model with this dataset type)
        sample_model = next(
            m for m in enabled_models if m.get("dataset", "normal") == dataset_type
        )
        dataset_files = get_dataset_files(sample_model)

        features_prepped_file = dataset_files["historical_file"].replace(
            ".parquet", "_with_neighbors.parquet"
        )

        print(f"Loading data from {features_prepped_file}...")
        df = pd.read_parquet(features_prepped_file)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values(by="date")
        df = df.loc[df["date"] >= START_DATE]
        df = df.loc[df["grade"] > data_config["min_grade"]]

        feature_cols = [
            col
            for col in df.columns
            if col not in exclude_cols and col not in bad_features
        ]

        load_and_prep_data(df, feature_cols, config, dataset_type)

    # Run hyperparameter search for all models
    split_grid_and_run_workers(config)
