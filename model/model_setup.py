import json
import os
import subprocess
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

FEATURES_PREPPED_FILE = "features_prepped.csv"
TRAIN_TEST_SPLIT = 0.8
VAL_TEST_SPLIT = 0.5
START_DATE = datetime(2025, 9, 8) + timedelta(days=28)
BAD_FEATURES = []
N_WORKERS = 4

PARAM_GRID = {
    'max_depth': [10, 12, 15, 20],
    'learning_rate': [0.05, 0.075, 0.1, 0.15, 0.2],
    'n_estimators': [10, 20, 50, 100],
    'min_child_weight': [5, 7, 10, 20],
    'subsample': [0.75, 0.8, 0.9],
    'colsample_bytree': [1.0],
    'gamma': [0],
    'reg_alpha': [0, 0.01, 0.1, 1],
    'reg_lambda': [0.5, 1, 2, 5],
    'colsample_bylevel': [0.7, 0.8, 1.0],
    'max_delta_step': [0, 1, 5],
}

def load_and_prep_data():
    print("Loading data...")
    if not os.path.exists(FEATURES_PREPPED_FILE):
        if os.path.exists(f"../{FEATURES_PREPPED_FILE}"):
             df = pd.read_csv(f"../{FEATURES_PREPPED_FILE}")
        else:
             raise FileNotFoundError(f"Could not find {FEATURES_PREPPED_FILE}")
    else:
        df = pd.read_csv(FEATURES_PREPPED_FILE)

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values(by='date')
    df = df[df['date'] >= START_DATE]
    
    feature_cols = [col for col in df.columns if col not in ['date', 'spec_id', 'price'] and col not in BAD_FEATURES]

    train_df = df.iloc[:int(len(df) * TRAIN_TEST_SPLIT)]
    test_df = df.iloc[int(len(df) * TRAIN_TEST_SPLIT):]
    val_df = test_df.iloc[:int(len(test_df) * VAL_TEST_SPLIT)]
    test_df = test_df.iloc[int(len(test_df) * VAL_TEST_SPLIT):]
    
    X_train = train_df[feature_cols].copy()
    y_train = train_df['price'].copy()
    
    X_val = val_df[feature_cols].copy()
    y_val = val_df['price'].copy()
    
    X_test = test_df[feature_cols].copy()
    y_test = test_df['price'].copy()
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    data_dir = "model/data"
    os.makedirs(data_dir, exist_ok=True)
    
    print("Saving datasets...")
    X_train.to_pickle(f"{data_dir}/X_train.pkl")
    y_train.to_pickle(f"{data_dir}/y_train.pkl")
    X_val.to_pickle(f"{data_dir}/X_val.pkl")
    y_val.to_pickle(f"{data_dir}/y_val.pkl")
    X_test.to_pickle(f"{data_dir}/X_test.pkl")
    y_test.to_pickle(f"{data_dir}/y_test.pkl")
    
    with open(f"{data_dir}/feature_cols.json", "w") as f:
        json.dump(feature_cols, f)

def split_grid_and_run_workers():
    full_grid = list(ParameterGrid(PARAM_GRID))
    np.random.shuffle(full_grid)
    total_params = len(full_grid)
    print(f"Total parameter combinations: {total_params}")
    
    chunk_size = int(np.ceil(total_params / N_WORKERS))
    chunks = [full_grid[i:i + chunk_size] for i in range(0, total_params, chunk_size)]
    
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
            "model/grid_search_worker.py", 
            "--gpu_id", str(i), 
            "--config", config_path
        ]
        
        p = subprocess.Popen(cmd)
        processes.append(p)
        print(f"Started worker {i} with {len(chunk)} combinations.")

    # Wait for all
    for p in processes:
        p.wait()
        
    print("All workers finished.")

if __name__ == "__main__":
    load_and_prep_data()
    split_grid_and_run_workers()
