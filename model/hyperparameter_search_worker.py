"""
Hyperparameter search worker.

This module runs hyperparameter search for a single model type on a single GPU.
It reads model configuration from model_config.json and uses custom loss functions
when specified.
"""

import argparse
import json
import os
import platform

import cupy as cp
import numpy as np
import pandas as pd
import xgboost

from tqdm import tqdm

from config_loader import (
    load_model_config,
    get_model_by_name,
    get_training_config,
    get_results_dir,
)


def weighted_loss(pred, dtrain, penalty_mult=10.0, penalize_overestimation=True):
    """
    Custom loss function for XGBoost that asymmetrically penalizes errors.

    Args:
        pred: Predicted values.
        dtrain: True values (DMatrix labels).
        penalty_mult: Multiplier for penalized direction.
        penalize_overestimation: If True, penalize overestimation; else underestimation.

    Returns:
        Tuple of (gradient, hessian) arrays.
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


def get_xgb_device(worker_id):
    """Determine best device for XGBoost."""
    try:
        cp.cuda.Device(int(worker_id)).use()
        print(f"Using CUDA (NVIDIA GPU) device {worker_id}")
        return f"cuda:{worker_id}"
    except Exception:
        pass

    if platform.processor() == "arm" or "Apple" in platform.processor():
        print("Using CPU with Apple Accelerate (M-series optimized)")
    else:
        print("Using CPU")

    return "cpu"


def load_data(data_dir):
    """Load pickled pandas data as numpy arrays."""
    print(f"Loading data from {data_dir}...")
    X_train = pd.read_parquet(os.path.join(data_dir, "X_train.parquet"))
    y_train = pd.read_pickle(os.path.join(data_dir, "y_train.pkl"))
    X_val = pd.read_parquet(os.path.join(data_dir, "X_val.parquet"))
    y_val = pd.read_pickle(os.path.join(data_dir, "y_val.pkl"))
    return X_train, y_train, X_val, y_val


def run_search(worker_id, config_path, model_name, dataset_type="normal"):
    """
    Run hyperparameter search for a specific model.

    Args:
        worker_id: GPU/worker ID.
        config_path: Path to the grid config JSON file.
        model_name: Name of the model to search for.
        dataset_type: The dataset type to use ("normal" or "log").
    """
    with open(config_path, "r") as f:
        param_grid = json.load(f)

    # Load model configuration
    full_config = load_model_config()
    model_config = get_model_by_name(full_config, model_name)
    training_config = get_training_config(full_config)

    if model_config is None:
        raise ValueError(f"Model '{model_name}' not found in configuration")

    objective = model_config["objective"]
    early_stopping_rounds = training_config.get("early_stopping_rounds", 10)

    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_type}")
    print(f"Objective: {objective}")

    data_dir = f"model/data/{dataset_type}"
    device = get_xgb_device(worker_id)

    X_train, y_train, X_val, y_val = load_data(data_dir)

    if "cuda" in device:
        X_train_gpu = cp.array(X_train)
        y_train_gpu = cp.array(y_train)
        X_val_gpu = cp.array(X_val)
        y_val_gpu = cp.array(y_val)
    else:
        X_train_gpu = X_train
        y_train_gpu = y_train
        X_val_gpu = X_val
        y_val_gpu = y_val

    best_score = float("inf")
    best_grid = {}

    results_dir = get_results_dir(model_name)
    os.makedirs(results_dir, exist_ok=True)
    result_file = os.path.join(results_dir, f"worker_{worker_id}_best.json")

    print(
        f"Worker {worker_id}: Starting hyperparameter search with {len(param_grid)} combinations."
    )

    for i, g in tqdm(
        enumerate(param_grid), total=len(param_grid), desc=f"Worker {worker_id}"
    ):
        try:
            # Build model parameters
            model_params = {
                "device": device,
                "n_jobs": -1,
                **g,
            }

            model_params["objective"] = objective

            es = xgboost.callback.EarlyStopping(
                rounds=early_stopping_rounds,
                min_delta=1e-3,
                save_best=True,
                maximize=False,
                data_name="validation_0",
            )

            model = xgboost.XGBRegressor(callbacks=[es])
            model.set_params(**model_params)
            model.fit(
                X_train_gpu,
                y_train_gpu,
                eval_set=[(X_val_gpu, y_val_gpu)],
                verbose=False,
            )
            # Extract native loss from XGBoost eval results
            evals_result = model.evals_result()
            val_metrics = evals_result["validation_0"]
            metric_name = list(val_metrics.keys())[0]
            best_n_estimators = model.best_iteration + 1
            native_loss = val_metrics[metric_name][model.best_iteration]

            g["n_estimators"] = best_n_estimators

            if native_loss < best_score:
                best_score = native_loss
                best_grid = g.copy()
                print(
                    f"\nWorker {worker_id}: New best {metric_name}: {native_loss:.6f} at iter {i}. n_estimators={best_n_estimators}"
                )

                result_data = {
                    "model_name": model_name,
                    "best_score": native_loss,
                    "metric_name": metric_name,
                    "best_params": best_grid,
                    "worker_id": worker_id,
                }
                with open(result_file, "w") as f:
                    json.dump(result_data, f, indent=2)

        except Exception as e:
            print(f"\nWorker {worker_id}: Error with params {g}: {e}")
            continue

    print(f"\nWorker {worker_id}: Finished. Best score: {best_score:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=str, required=True, help="Worker/GPU ID")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to grid config json"
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name of the model to train"
    )
    parser.add_argument(
        "--dataset", type=str, default="normal", help="Dataset type (normal or log)"
    )

    args = parser.parse_args()

    run_search(args.gpu_id, args.config, args.model_name, args.dataset)
