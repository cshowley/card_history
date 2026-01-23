import glob
import json
import multiprocessing
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
import constants
from data_integrity import get_tracker


LOWER_QUANTILE = 0.05
UPPER_QUANTILE = 0.95


def weighted_loss(pred, dtrain, penalty_mult=10.0, penalize_overestimation=True):
    """
    Custom loss function for XGBoost.
    """
    y_true = dtrain
    y_pred = pred

    residual = y_pred - y_true
    grad = residual.copy()
    hess = np.ones_like(residual)

    if penalize_overestimation:
        mask = residual > 0
        grad[mask] *= penalty_mult
        hess[mask] *= penalty_mult
    else:
        mask = residual < 0
        grad[mask] *= penalty_mult
        hess[mask] *= penalty_mult

    return grad, hess


def lower_bound_loss(y_true, y_pred):
    return weighted_loss(
        y_pred, y_true, penalty_mult=10.0, penalize_overestimation=True
    )


def upper_bound_loss(y_true, y_pred):
    return weighted_loss(
        y_pred, y_true, penalty_mult=10.0, penalize_overestimation=False
    )


def calculate_validation_mdape(input_file, best_params, tracker):
    """
    Trains a temporary model on a time-based train split and calculates
    Median Absolute Percentage Error on the validation set for benchmarking.
    The model is discarded after metric calculation.
    """
    print("Calculating validation MdAPE (time-based split)...")

    df = pd.read_parquet(input_file)
    df = df.dropna(subset=["price"])
    df["grade"] = pd.to_numeric(df["grade"], errors="raise")
    df["half_grade"] = pd.to_numeric(df["half_grade"], errors="raise")
    df["seller_popularity"] = pd.to_numeric(df["seller_popularity"], errors="raise")
    df = df[df.date >= datetime(2025, 9, 1)]

    # Time-based split: sort by date and use most recent 20% for validation
    df = df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    print(f"  Train set: {len(train_df)} samples, Val set: {len(val_df)} samples")
    print(f"  Train dates: {train_df['date'].min()} to {train_df['date'].max()}")
    print(f"  Val dates: {val_df['date'].min()} to {val_df['date'].max()}")

    exclude_cols = ["gemrate_id", "date", "price", "_row_id"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X_train = train_df[feature_cols]
    y_train = train_df["price"]
    X_val = val_df[feature_cols]
    y_val = val_df["price"]

    # Train temporary model
    val_params = {k: v for k, v in best_params.items() if k != "objective"}
    val_params["device"] = "cpu"  # Use CPU for validation model
    temp_model = xgb.XGBRegressor(objective="reg:gamma")
    temp_model.set_params(**val_params)
    temp_model.fit(X_train, y_train)

    # Predict on validation set
    y_pred = temp_model.predict(X_val)

    # Calculate Median Absolute Percentage Error
    ape = np.abs(y_val.values - y_pred) / y_val.values * 100
    mdape = np.median(ape)
    mape = np.mean(ape)

    print(f"  Validation MdAPE: {mdape:.2f}%")
    print(f"  Validation MAPE: {mape:.2f}%")

    tracker.add_metric(
        id="s7_validation_mdape",
        title="Validation MdAPE",
        value=f"{mdape:.2f}%",
    )
    tracker.add_metric(
        id="s7_validation_mape",
        title="Validation MAPE",
        value=f"{mape:.2f}%",
    )

    # Discard the temporary model
    del temp_model
    return mdape


def get_best_params():
    """Finds the best hyperparameters from the model/results directory."""
    results_dir = "model/results"
    result_files = glob.glob(os.path.join(results_dir, "worker_*_best.json"))

    if not result_files:
        raise FileNotFoundError(f"No result files found in {results_dir}")

    best_mape = float("inf")
    best_params = None
    best_worker = None

    for file in result_files:
        with open(file, "r") as f:
            data = json.load(f)
            if "best_mape" in data and "best_params" in data:
                if data["best_mape"] < best_mape:
                    best_mape = data["best_mape"]
                    best_params = data["best_params"]
                    best_worker = data.get("worker_id", "unknown")
    if best_params is None:
        raise ValueError("Could not find any valid best parameters in result files.")

    print(f"Best params found from worker {best_worker} with mape: {best_mape:.4f}")
    return best_params


def train_worker(config, best_params, input_file, gpu_id):
    """
    Worker function to train a single XGBoost model on a specific GPU.
    """
    try:
        print(f"[{config['name']}] Starting training on GPU {gpu_id}...")

        print(f"[{config['name']}] Loading data...")
        df = pd.read_parquet(input_file)
        df = df.dropna(subset=["price"])

        df["grade"] = pd.to_numeric(df["grade"], errors="raise")
        df["half_grade"] = pd.to_numeric(df["half_grade"], errors="raise")
        df["seller_popularity"] = pd.to_numeric(df["seller_popularity"], errors="raise")
        df = df[df.date >= datetime(2025, 9, 1)]
        exclude_cols = ["gemrate_id", "date", "price", "_row_id"]
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        X = df[feature_cols]
        y = df["price"]

        print(f"[{config['name']}] Training on {len(X)} samples...")

        worker_params = {k: v for k, v in best_params.items() if k != "objective"}
        worker_params["device"] = f"cuda:{gpu_id}"
        model = xgb.XGBRegressor(objective=config["objective"])
        model.set_params(**worker_params)
        model.fit(X, y)

        print(f"[{config['name']}] Saving model to {config['file']}...")
        model.save_model(config["file"])
        print(f"[{config['name']}] Finished successfully.")

    except Exception as e:
        print(f"[{config['name']}] Failed: {e}")
        raise e


def train_model():
    """Trains the XGBoost models in parallel using multiprocessing."""
    print("Step 7: Training XGBoost Models Parallelly")
    start_time = time.time()
    tracker = get_tracker()

    input_file = constants.S3_HISTORICAL_DATA_FILE.replace(
        ".parquet", "_with_neighbors.parquet"
    )
    best_params = get_best_params()

    # Calculate validation MdAPE for benchmarking (before full training)
    calculate_validation_mdape(input_file, best_params, tracker)

    models_config = [
        {
            "name": "Gamma",
            "file": constants.S7_OUTPUT_MODEL_FILE.replace(".json", "_gamma.json"),
            "objective": "reg:gamma",
        },
        {
            "name": "Lower",
            "file": constants.S7_OUTPUT_MODEL_FILE.replace(".json", "_lower.json"),
            "objective": lower_bound_loss,
        },
        {
            "name": "Upper",
            "file": constants.S7_OUTPUT_MODEL_FILE.replace(".json", "_upper.json"),
            "objective": upper_bound_loss,
        },
    ]

    processes = []
    for i, config in enumerate(models_config):
        gpu_id = i % 4
        p = multiprocessing.Process(
            target=train_worker, args=(config, best_params, input_file, gpu_id)
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    for p in processes:
        if p.exitcode != 0:
            raise RuntimeError("One or more training processes failed.")

    # Data Integrity Tracking
    duration = time.time() - start_time
    tracker.add_metric(
        id="s7_models_trained",
        title="Models Trained",
        value="3 (Gamma, Lower, Upper)",
    )
    tracker.add_metric(
        id="s7_duration",
        title="Step 7 Duration",
        value=f"{duration:.1f}s",
    )

    print("All models trained and saved successfully.")


def run_step_7():
    try:
        try:
            multiprocessing.set_start_method("spawn")
        except RuntimeError:
            pass

        train_model()
    except Exception as e:
        print(f"Failed to run Step 7: {e}")
        raise e


if __name__ == "__main__":
    run_step_7()
