import argparse
import json
import os
import platform
import numpy as np
import pandas as pd
from tqdm import tqdm
from xgboost import XGBRegressor

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


def mdape(y_true, y_pred):
    ape = np.abs((y_true - y_pred) / y_true)
    return np.median(ape)


def get_xgb_device(worker_id):
    """Determine best device for XGBoost."""
    try:
        import cupy as cp

        cp.cuda.Device(worker_id).use()
        print("Using CUDA (NVIDIA GPU)")
        return f"cuda:{worker_id}"
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
    X_train = pd.read_parquet(os.path.join(data_dir, "X_train.parquet"))
    y_train = pd.read_pickle(os.path.join(data_dir, "y_train.pkl"))
    X_val = pd.read_parquet(os.path.join(data_dir, "X_val.parquet"))
    y_val = pd.read_pickle(os.path.join(data_dir, "y_val.pkl"))
    return X_train, y_train, X_val, y_val


def run_search(worker_id, config_path):
    with open(config_path, "r") as f:
        param_grid = json.load(f)

    data_dir = "model/data"
    device = get_xgb_device(worker_id)

    X_train, y_train, X_val, y_val = load_data(data_dir)

    best_score = float("inf")
    best_grid = {}

    results_dir = "model/results"
    os.makedirs(results_dir, exist_ok=True)
    result_file = os.path.join(results_dir, f"worker_{worker_id}_best.json")

    print(
        f"Worker {worker_id}: Starting hyperparameter search with {len(param_grid)} combinations."
    )

    for i, g in tqdm(
        enumerate(param_grid), total=len(param_grid), desc=f"Worker {worker_id}"
    ):
        try:
            model_upper = XGBRegressor(device=device, objective=upper_bound_loss, n_jobs=-1)
            model_upper.set_params(**g)
            model_upper.fit(X_train, y_train, verbose=False)

            y_val_pred_upper = model_upper.predict(X_val)

            model_lower = XGBRegressor(device=device, objective=lower_bound_loss, n_jobs=-1)
            model_lower.set_params(**g)
            model_lower.fit(X_train, y_train, verbose=False)

            y_val_pred_lower = model_upper.predict(X_val)

            y_val_pred = (y_val_pred_upper + y_val_pred_lower) / 2.0

            def to_cpu_numpy(x):
                if isinstance(x, (pd.DataFrame, pd.Series)):
                    return x.to_numpy()
                if hasattr(x, "get"):
                    return x.get()
                if hasattr(x, "to_numpy"):
                    return x.to_numpy()
                return np.array(x)

            y_val_np = to_cpu_numpy(y_val)
            y_val_pred_np = to_cpu_numpy(y_val_pred)

            mape = mdape(np.exp(y_val_np), np.exp(y_val_pred_np))

            if mape < best_score:
                best_score = mape
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
