import argparse
import json
import os
import platform

import numpy as np
import pandas as pd

from tqdm import tqdm
from xgboost import XGBRegressor


def get_xgb_device():
    """Determine best device for XGBoost."""
    try:
        import cupy as cp

        cp.cuda.Device(0).compute_capability
        print("Using CUDA (NVIDIA GPU)")
        return "cuda:0"
    except:  # noqa: E722
        pass

    if platform.processor() == "arm" or "Apple" in platform.processor():
        print("Using CPU with Apple Accelerate (M-series optimized)")
    else:
        print("Using CPU")

    return "cpu"


def load_data(data_dir):
    """Load pickled pandas data as numpy arrays."""
    print(f"Loading data from {data_dir}...")
    X_train = pd.read_pickle(os.path.join(data_dir, "X_train.pkl")).to_numpy()
    y_train = pd.read_pickle(os.path.join(data_dir, "y_train.pkl")).to_numpy().ravel()
    X_val = pd.read_pickle(os.path.join(data_dir, "X_val.pkl")).to_numpy()
    y_val = pd.read_pickle(os.path.join(data_dir, "y_val.pkl")).to_numpy().ravel()
    return X_train, y_train, X_val, y_val


def run_search(worker_id, config_path):
    with open(config_path, "r") as f:
        param_grid = json.load(f)

    data_dir = "model/data"
    device = get_xgb_device()

    X_train, y_train, X_val, y_val = load_data(data_dir)

    X_train = np.nan_to_num(X_train, nan=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0)

    best_score = float("inf")
    best_grid = {}

    results_dir = "model/results"
    os.makedirs(results_dir, exist_ok=True)
    result_file = os.path.join(results_dir, f"worker_{worker_id}_best.json")

    print(
        f"Worker {worker_id}: Starting grid search with {len(param_grid)} combinations."
    )

    model = XGBRegressor(device=device, n_jobs=-1)

    for i, g in tqdm(
        enumerate(param_grid), total=len(param_grid), desc=f"Worker {worker_id}"
    ):
        try:
            model.set_params(**g)
            model.fit(X_train, y_train, verbose=False)
            y_val_pred = model.predict(X_val)
            ape = np.abs((y_val - y_val_pred) / y_val)
            mdape = np.median(ape)

            if mdape < best_score:
                best_score = mdape
                best_grid = g
                print(
                    f"\nWorker {worker_id}: New best MdAPE: {best_score:.2%} at iter {i}"
                )

                result_data = {
                    "best_mape": best_score,
                    "best_params": best_grid,
                    "worker_id": worker_id,
                }
                with open(result_file, "w") as f:
                    json.dump(result_data, f, indent=2)

        except Exception as e:
            print(f"\nWorker {worker_id}: Error with params {g}: {e}")
            continue

    print(f"\nWorker {worker_id}: Finished. Best MdAPE: {best_score:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=str, required=True, help="Worker ID")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to grid config json"
    )

    args = parser.parse_args()

    run_search(args.gpu_id, args.config)
