import argparse
import json
import os
import pandas as pd
from tqdm import tqdm
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error

def load_data(data_dir):
    print(f"Loading data from {data_dir}...")
    X_train = pd.read_pickle(os.path.join(data_dir, "X_train.pkl"))
    y_train = pd.read_pickle(os.path.join(data_dir, "y_train.pkl"))
    X_val = pd.read_pickle(os.path.join(data_dir, "X_val.pkl"))
    y_val = pd.read_pickle(os.path.join(data_dir, "y_val.pkl"))
    return X_train, y_train, X_val, y_val

def run_search(gpu_id, config_path):
    with open(config_path, 'r') as f:
        param_grid = json.load(f)

    data_dir = "model/data" 
    
    X_train, y_train, X_val, y_val = load_data(data_dir)
    
    best_score = float('inf')
    best_grid = {}
    
    results_dir = "model/results"
    os.makedirs(results_dir, exist_ok=True)
    result_file = os.path.join(results_dir, f"worker_{gpu_id}_best.json")

    print(f"Worker {gpu_id}: Starting grid search with {len(param_grid)} combinations.")
    
    device_arg = f"cuda:{gpu_id}"
    
    model = XGBRegressor(device=device_arg)
    
    for i, g in tqdm(enumerate(param_grid)):
        try:
            model.set_params(**g)
            model.fit(X_train, y_train)
            y_val_pred = model.predict(X_val)
            mape = mean_absolute_percentage_error(y_val, y_val_pred)
            
            if mape < best_score:
                best_score = mape
                best_grid = g
                print(f"Worker {gpu_id}: New best MAPE: {best_score:.2%} at iter {i}")
                
                result_data = {
                    "best_mape": best_score,
                    "best_params": best_grid,
                    "gpu_id": gpu_id
                }
                with open(result_file, "w") as f:
                    json.dump(result_data, f, indent=2)
            
            if i % 10 == 0:
                print(f"Worker {gpu_id}: Processed {i}/{len(param_grid)}...")
                
        except Exception as e:
            print(f"Worker {gpu_id}: Error with params {g}: {e}")
            continue

    print(f"Worker {gpu_id}: Finished. Best MAPE: {best_score:.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=str, required=True, help="GPU ID to use")
    parser.add_argument("--config", type=str, required=True, help="Path to grid config json")
    
    args = parser.parse_args()
    
    run_search(args.gpu_id, args.config)
