import concurrent.futures
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb
from tqdm import tqdm

import constants


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

    model_file = constants.S7_OUTPUT_MODEL_FILE
    input_file = constants.S2_TODAY_DATA_FILE.replace(
        ".parquet", "_with_neighbors.parquet"
    )
    output_file = constants.S8_PREDICTIONS_FILE

    # Check that both model files exist
    model_lower_file = model_file.replace(".json", "_lower.json")
    model_upper_file = model_file.replace(".json", "_upper.json")
    
    if not os.path.exists(model_lower_file):
        raise FileNotFoundError(f"Lower model file not found: {model_lower_file}")
    if not os.path.exists(model_upper_file):
        raise FileNotFoundError(f"Upper model file not found: {model_upper_file}")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    print("Loading models on separate GPUs...")
    models = {}

    model_lower_file = model_file.replace(".json", "_lower.json")
    print(f"Loading Lower model from {model_lower_file} on cuda:1...")
    model_lower = xgb.XGBRegressor()
    model_lower.load_model(model_lower_file)
    model_lower.set_params(device="cuda:0")
    models["prediction_lower"] = (model_lower, "cuda:0")

    model_upper_file = model_file.replace(".json", "_upper.json")
    print(f"Loading Upper model from {model_upper_file} on cuda:2...")
    model_upper = xgb.XGBRegressor()
    model_upper.load_model(model_upper_file)
    model_upper.set_params(device="cuda:1")
    models["prediction_upper"] = (model_upper, "cuda:1")

    print(f"Processing {input_file} in chunks...")

    parquet_file = pq.ParquetFile(input_file)

    writer = None
    total_rows = 0
    batch_size = constants.S6_BATCH_SIZE

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
                batch_results[col_name] = np.exp(preds)
            except Exception as e:
                print(f"Prediction failed for {col_name}: {e}")
                raise e

        for col_name in models.keys():
            ident_df[col_name] = batch_results[col_name]

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


if __name__ == "__main__":
    run_step_8()
