import concurrent.futures
import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb
from tqdm import tqdm

import constants
from data_integrity import get_tracker


def reverse_dummies(df, prefix, drop_prefix=True):
    """
    Reconstructs the original categorical column from dummy columns.
    Assumes dummy columns start with 'prefix_'.
    """
    dummy_cols = [c for c in df.columns if c.startswith(prefix + "_")]
    if not dummy_cols:
        return pd.Series(index=df.index, dtype=object)

    def get_col_name(row):
        for c in dummy_cols:
            if row[c] == 1:
                return c
        return None

    subset = df[dummy_cols]
    max_col = subset.idxmax(axis=1)
    clean_vals = max_col.str.replace(prefix + "_", "")
    mask = subset.max(axis=1) == 1
    clean_vals = clean_vals.where(mask, "Unknown")
    return clean_vals


def predict_worker(model, X, device):
    """
    Helper to run prediction.
    """
    return model.predict(X)


def run_step_8():
    print("Starting Step 8: Inference on Today's Data Parallelly...")
    start_time = time.time()
    tracker = get_tracker()

    model_file = constants.S7_OUTPUT_MODEL_FILE
    input_file = constants.S3_TODAY_DATA_FILE.replace(
        ".parquet", "_with_neighbors.parquet"
    )
    output_file = constants.S8_PREDICTIONS_FILE

    # Check all three model files exist (step_7 creates _lower, _upper, _gamma versions)
    for suffix in ["_lower.json", "_upper.json", "_gamma.json"]:
        file_path = model_file.replace(".json", suffix)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    print("Loading models on separate GPUs...")
    models = {}

    model_lower_file = model_file.replace(".json", "_lower.json")
    print(f"Loading Lower model from {model_lower_file} on cuda:0...")
    model_lower = xgb.XGBRegressor()
    model_lower.load_model(model_lower_file)
    model_lower.set_params(device="cuda:0")
    models["prediction_lower"] = (model_lower, "cuda:0")

    model_upper_file = model_file.replace(".json", "_upper.json")
    print(f"Loading Upper model from {model_upper_file} on cuda:1...")
    model_upper = xgb.XGBRegressor()
    model_upper.load_model(model_upper_file)
    model_upper.set_params(device="cuda:1")
    models["prediction_upper"] = (model_upper, "cuda:1")

    model_gamma_file = model_file.replace(".json", "_gamma.json")
    print(f"Loading Gamma model from {model_gamma_file} on cuda:2...")
    model_gamma = xgb.XGBRegressor()
    model_gamma.load_model(model_gamma_file)
    model_gamma.set_params(device="cuda:2")
    models["prediction"] = (model_gamma, "cuda:2")

    print(f"Processing {input_file} in chunks...")

    parquet_file = pq.ParquetFile(input_file)

    writer = None
    total_rows = 0
    batch_size = constants.S6_BATCH_SIZE

    # Prediction stats accumulators
    all_predictions = []
    all_predictions_lower = []
    all_predictions_upper = []
    negative_predictions_count = 0

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)

    for batch in tqdm(
        parquet_file.iter_batches(batch_size=batch_size), desc="Running Inference"
    ):
        df = batch.to_pandas()

        exclude_cols = ["gemrate_id", "date", "price", "_row_id"]
        ident_df = df[["gemrate_id", "grade", "half_grade"]].copy()
        conditions_gc = [
            df["grade_co_PSA"] == 1,
            df["grade_co_BGS"] == 1,
            df["grade_co_CGC"] == 1,
        ]
        choices_gc = ["PSA", "BGS", "CGC"]
        ident_df["grading_company"] = np.select(
            conditions_gc, choices_gc, default="Unknown"
        )
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        X = df[feature_cols]

        future_map = {}
        for col_name, (model, device) in models.items():
            future = executor.submit(predict_worker, model, X, device)
            future_map[future] = col_name

        batch_results = {}
        for future in concurrent.futures.as_completed(future_map):
            col_name = future_map[future]
            try:
                preds = future.result()
                batch_results[col_name] = preds
            except Exception as e:
                print(f"Prediction failed for {col_name}: {e}")
                raise e

        for col_name in models.keys():
            ident_df[col_name] = batch_results[col_name]

        # Accumulate prediction stats
        if "prediction" in batch_results:
            all_predictions.extend(batch_results["prediction"].tolist())
        if "prediction_lower" in batch_results:
            all_predictions_lower.extend(batch_results["prediction_lower"].tolist())
        if "prediction_upper" in batch_results:
            all_predictions_upper.extend(batch_results["prediction_upper"].tolist())
        if "prediction" in batch_results:
            negative_predictions_count += int(np.sum(batch_results["prediction"] < 0))

        table = pa.Table.from_pandas(ident_df)

        if writer is None:
            writer = pq.ParquetWriter(output_file, table.schema)

        writer.write_table(table)
        total_rows += len(ident_df)

    executor.shutdown()

    if writer:
        writer.close()
        print(f"Inference complete. Saved {total_rows} predictions to {output_file}.")
    else:
        print("No data processed.")

    # Data Integrity Tracking
    duration = time.time() - start_time
    tracker.add_metric(
        id="s8_predictions_generated",
        title="Predictions Generated",
        value=total_rows,
    )

    # Prediction distribution stats
    if all_predictions:
        preds_arr = np.array(all_predictions)
        pred_min = float(np.min(preds_arr))
        pred_max = float(np.max(preds_arr))
        pred_mean = float(np.mean(preds_arr))
        pred_median = float(np.median(preds_arr))
        pred_std = float(np.std(preds_arr))

        tracker.add_metric(
            id="s8_prediction_min",
            title="Prediction Min",
            value=round(pred_min, 4),
        )
        tracker.add_metric(
            id="s8_prediction_max",
            title="Prediction Max",
            value=round(pred_max, 4),
        )
        tracker.add_metric(
            id="s8_prediction_mean",
            title="Prediction Mean",
            value=round(pred_mean, 4),
        )
        tracker.add_metric(
            id="s8_prediction_median",
            title="Prediction Median",
            value=round(pred_median, 4),
        )
        tracker.add_metric(
            id="s8_prediction_std",
            title="Prediction Std Dev",
            value=round(pred_std, 4),
        )
        tracker.add_metric(
            id="s8_negative_predictions_count",
            title="Negative Predictions Count",
            value=negative_predictions_count,
        )

        # Model collapse detection
        if pred_min == pred_max:
            tracker.add_error(
                f"Model collapse detected: all predictions are {pred_min}",
                step="step_8",
            )
            print(f"ERROR: Model collapse detected! All predictions = {pred_min}")

        if negative_predictions_count > 0:
            print(f"WARNING: {negative_predictions_count} negative predictions detected.")

    if all_predictions_lower:
        lower_arr = np.array(all_predictions_lower)
        tracker.add_metric(
            id="s8_prediction_lower_min",
            title="Prediction Lower Min",
            value=round(float(np.min(lower_arr)), 4),
        )
        tracker.add_metric(
            id="s8_prediction_lower_max",
            title="Prediction Lower Max",
            value=round(float(np.max(lower_arr)), 4),
        )

    if all_predictions_upper:
        upper_arr = np.array(all_predictions_upper)
        tracker.add_metric(
            id="s8_prediction_upper_min",
            title="Prediction Upper Min",
            value=round(float(np.min(upper_arr)), 4),
        )
        tracker.add_metric(
            id="s8_prediction_upper_max",
            title="Prediction Upper Max",
            value=round(float(np.max(upper_arr)), 4),
        )

    tracker.add_metric(
        id="s8_duration",
        title="Step 8 Duration",
        value=round(duration, 1),
    )


if __name__ == "__main__":
    run_step_8()
