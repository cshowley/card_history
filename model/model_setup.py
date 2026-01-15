import json
import math
import os
import random
import subprocess
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

FEATURES_PREPPED_FILE = "historical_data.parquet"
TRAIN_TEST_SPLIT = 0.8
VAL_TEST_SPLIT = 0.5
START_DATE = datetime(2025, 9, 8) + timedelta(days=28)
BAD_FEATURES = []
N_WORKERS = 2
N_TRIALS = 4000

PARAM_RANGES = {
    "max_depth": (3, 15),
    "learning_rate": (0.01, 0.2),
    "n_estimators": (100, 2000),
    "min_child_weight": (1, 7),
    "subsample": (0.6, 0.95),
    "colsample_bytree": (0.6, 0.95),
    "gamma": (0.0, 5.0),
    "reg_alpha": (0.0, 10.0),
    "reg_lambda": (1.0, 10.0),
}


def load_and_prep_data(df, feature_cols):
    train_df = df.iloc[: int(len(df) * TRAIN_TEST_SPLIT)]
    test_df = df.iloc[int(len(df) * TRAIN_TEST_SPLIT) :]
    val_df = test_df.iloc[: int(len(test_df) * VAL_TEST_SPLIT)]
    test_df = test_df.iloc[int(len(test_df) * VAL_TEST_SPLIT) :]

    X_train = train_df[feature_cols]
    y_train = train_df["price"]

    X_val = val_df[feature_cols]
    y_val = val_df["price"]

    X_test = test_df[feature_cols]
    y_test = test_df["price"]

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    data_dir = "model/data"
    os.makedirs(data_dir, exist_ok=True)

    print("Saving datasets...")
    X_train.to_parquet(f"{data_dir}/X_train.parquet")
    y_train.to_pickle(f"{data_dir}/y_train.pkl")
    X_val.to_parquet(f"{data_dir}/X_val.parquet")
    y_val.to_pickle(f"{data_dir}/y_val.pkl")
    X_test.to_parquet(f"{data_dir}/X_test.parquet")
    y_test.to_pickle(f"{data_dir}/y_test.pkl")

    with open(f"{data_dir}/feature_cols.json", "w") as f:
        json.dump(feature_cols, f)


def get_random_params():
    params = {}

    params["max_depth"] = random.randint(*PARAM_RANGES["max_depth"])
    params["n_estimators"] = random.randint(*PARAM_RANGES["n_estimators"])
    params["min_child_weight"] = random.randint(*PARAM_RANGES["min_child_weight"])

    params["subsample"] = random.uniform(*PARAM_RANGES["subsample"])
    params["colsample_bytree"] = random.uniform(*PARAM_RANGES["colsample_bytree"])

    lr_min, lr_max = PARAM_RANGES["learning_rate"]
    params["learning_rate"] = math.exp(
        random.uniform(math.log(lr_min), math.log(lr_max))
    )

    def log_uniform(min_val, max_val):
        min_val = max(min_val, 1e-9)
        return math.exp(random.uniform(math.log(min_val), math.log(max_val)))

    params["gamma"] = log_uniform(*PARAM_RANGES["gamma"])
    params["reg_alpha"] = log_uniform(*PARAM_RANGES["reg_alpha"])
    params["reg_lambda"] = log_uniform(*PARAM_RANGES["reg_lambda"])
    return params


def split_grid_and_run_workers(feature_cols):
    print(f"Generating {N_TRIALS} random parameter combinations...")

    random_search_grid = [get_random_params() for _ in range(N_TRIALS)]

    # for params in random_search_grid:
    #     params["monotone_constraints"] = {"grade": 1, "half_grade": 1}

    chunk_size = int(np.ceil(N_TRIALS / N_WORKERS))
    chunks = [
        random_search_grid[i : i + chunk_size] for i in range(0, N_TRIALS, chunk_size)
    ]

    config_dir = "model/configs"
    os.makedirs(config_dir, exist_ok=True)

    processes = []

    print("Launching workers...")
    for i, chunk in enumerate(chunks):
        if not chunk:
            continue

        config_path = f"{config_dir}/grid_chunk_{i}.json"
        with open(config_path, "w") as f:
            json.dump(chunk, f)

        cmd = [
            sys.executable,
            "model/hyperparameter_search_worker.py",
            "--gpu_id",
            str(i),
            "--config",
            config_path,
        ]

        p = subprocess.Popen(cmd)
        processes.append(p)
        print(f"Started worker {i} with {len(chunk)} combinations.")

    for p in processes:
        p.wait()

    print("All workers finished.")


if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_parquet(FEATURES_PREPPED_FILE)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(by="date")
    df = df.loc[df["date"] >= START_DATE]
    df = df.loc[df["grade"] > 6]

    feature_cols = [
        col
        for col in df.columns
        if col not in ["gemrate_id", "date", "price"] and col not in BAD_FEATURES
    ]
    load_and_prep_data(df, feature_cols)
    split_grid_and_run_workers(feature_cols)
