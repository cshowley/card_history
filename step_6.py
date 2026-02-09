import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

import constants
from data_integrity import get_tracker


def s6_load_index_series():
    """Load index data and return a date -> index_value series."""
    if not os.path.exists(constants.S1_INDEX_FILE):
        print(
            f"Warning: {constants.S1_INDEX_FILE} not found. Index prices will be NaN."
        )
        return None
    idx_df = pd.read_csv(constants.S1_INDEX_FILE)
    idx_df["date"] = pd.to_datetime(idx_df["date"])
    idx_df = idx_df.groupby("date")["index_value"].mean()
    return idx_df


def s6_process_chunk(chunk_df, df_neighbors, neighbor_sales, index_series):
    relevant_ids = chunk_df["gemrate_id"].unique()
    chunk_neighbors = df_neighbors[df_neighbors["gemrate_id"].isin(relevant_ids)].copy()
    merged = chunk_df[["_row_id", "gemrate_id", "grade", "date"]].merge(
        chunk_neighbors[["gemrate_id", "neighbor_id", "score", "neighbor_rank"]],
        on="gemrate_id",
        how="left",
    )

    merged = merged.merge(neighbor_sales, on=["neighbor_id", "grade"], how="left")

    merged = merged[merged["sale_date"] < merged["date"]]

    if merged.empty:
        return pd.DataFrame({"_row_id": chunk_df["_row_id"]})

    merged = merged.sort_values(
        by=["_row_id", "neighbor_rank", "sale_date"], ascending=[True, True, False]
    )

    merged["dense_neighbor_rank"] = merged.groupby("_row_id")[
        "neighbor_rank"
    ].transform(lambda x: pd.factorize(x, sort=True)[0] + 1)

    merged = merged[merged["dense_neighbor_rank"] <= constants.S6_N_NEIGHBORS]
    merged["sale_rank"] = (
        merged.groupby(["_row_id", "dense_neighbor_rank"]).cumcount() + 1
    )
    merged = merged[merged["sale_rank"] <= constants.S6_N_NEIGHBOR_SALES]

    merged["ndays"] = (merged["date"] - merged["sale_date"]).dt.days
    if index_series is not None:
        merged["index_price"] = merged["sale_date"].map(index_series)
    else:
        merged["index_price"] = np.nan

    merged["pivot_key"] = (
        merged["dense_neighbor_rank"].astype(str)
        + "_"
        + merged["sale_rank"].astype(str)
    )

    score_subset = merged[["_row_id", "dense_neighbor_rank", "score"]].drop_duplicates(
        ["_row_id", "dense_neighbor_rank"]
    )
    pivot_score = score_subset.pivot(
        index="_row_id", columns="dense_neighbor_rank", values="score"
    )

    pivot_price = merged.pivot(
        index="_row_id", columns="pivot_key", values="sale_price"
    )
    pivot_ndays = merged.pivot(index="_row_id", columns="pivot_key", values="ndays")
    pivot_index = merged.pivot(
        index="_row_id", columns="pivot_key", values="index_price"
    )

    feature_data = {}

    for n in range(1, constants.S6_N_NEIGHBORS + 1):
        col_score = f"neighbor_{n}_score"
        feature_data[col_score] = pivot_score[n] if n in pivot_score.columns else np.nan

        for s in range(1, constants.S6_N_NEIGHBOR_SALES + 1):
            key = f"{n}_{s}"
            col_price = f"neighbor_{n}_sale_{s}_price"
            col_ndays = f"neighbor_{n}_sale_{s}_ndays"
            col_idx = f"neighbor_{n}_sale_{s}_index_price"

            feature_data[col_price] = (
                pivot_price[key] if key in pivot_price.columns else np.nan
            )
            feature_data[col_ndays] = (
                pivot_ndays[key] if key in pivot_ndays.columns else np.nan
            )
            feature_data[col_idx] = (
                pivot_index[key] if key in pivot_index.columns else np.nan
            )

    chunk_result = pd.DataFrame(feature_data)
    chunk_result = pd.DataFrame(feature_data)
    chunk_result = chunk_result.reset_index()
    return chunk_result


def run_step_6():
    print("Starting Step 6: Neighbor Features...")
    start_time = time.time()
    tracker = get_tracker()

    sales_cols = ["gemrate_id", "grade", "date", "price"]

    df_sales_lookup = pd.read_parquet(
        constants.S3_HISTORICAL_DATA_FILE, columns=sales_cols
    )
    df_sales_lookup["date"] = pd.to_datetime(
        df_sales_lookup["date"], format="mixed", errors="coerce"
    )
    df_sales_lookup = df_sales_lookup.dropna(subset=["date"])
    df_sales_lookup = df_sales_lookup.reset_index(drop=True)

    print("Loading neighbors...")
    df_neighbors = pd.read_parquet(constants.S5_OUTPUT_NEIGHBORS_FILE)
    df_neighbors = df_neighbors.rename(columns={"neighbors": "neighbor_id"})

    df_neighbors["neighbor_rank"] = df_neighbors.groupby("gemrate_id").cumcount() + 1

    print("Loading index data...")
    index_series = s6_load_index_series()

    print("Pre-preparing neighbor sales lookup...")
    df_sales_lookup = df_sales_lookup.rename(
        columns={
            "gemrate_id": "neighbor_id",
            "date": "sale_date",
            "price": "sale_price",
        }
    )
    df_sales_lookup = df_sales_lookup.dropna()
    df_sales_lookup["neighbor_id"] = df_sales_lookup["neighbor_id"].astype(str)

    file_row_counts = {}
    for file in [constants.S3_HISTORICAL_DATA_FILE, constants.S3_TODAY_DATA_FILE]:
        output_file = file.replace(".parquet", "_with_neighbors.parquet")

        print(f"Processing {file}...")

        try:
            parquet_file = pq.ParquetFile(file)
        except Exception as e:
            print(f"Could not open {file}: {e}")
            continue

        writer = None
        total_rows = 0
        batch_size = constants.S6_BATCH_SIZE

        for batch in tqdm(
            parquet_file.iter_batches(batch_size=batch_size),
            desc=f"Processing {os.path.basename(file)}",
        ):
            chunk_df = batch.to_pandas()

            chunk_df["date"] = pd.to_datetime(
                chunk_df["date"], format="mixed", errors="coerce"
            )
            chunk_df = chunk_df.dropna(subset=["date"])

            if chunk_df.empty:
                continue

            chunk_df = chunk_df.reset_index(drop=True)
            chunk_df["_row_id"] = chunk_df.index

            chunk_features = s6_process_chunk(
                chunk_df, df_neighbors, df_sales_lookup, index_series
            )

            chunk_result = chunk_df.merge(chunk_features, on="_row_id", how="left")
            chunk_result.drop(columns=["_row_id"], inplace=True)

            table = pa.Table.from_pandas(chunk_result)

            if writer is None:
                writer = pq.ParquetWriter(output_file, table.schema)

            writer.write_table(table)
            total_rows += len(chunk_result)

            del chunk_df, chunk_features, chunk_result, table

        if writer:
            writer.close()
            print(
                f"Adding neighbors to {file} complete. Saved {total_rows} rows to {output_file}."
            )
        else:
            print(f"No data processed for {file}")

        file_row_counts[os.path.basename(file)] = total_rows

    # Data Integrity Tracking
    duration = time.time() - start_time
    hist_basename = os.path.basename(constants.S3_HISTORICAL_DATA_FILE)
    today_basename = os.path.basename(constants.S3_TODAY_DATA_FILE)
    tracker.add_metric(
        id="s6_historical_rows_output",
        title="Historical Rows with Neighbors",
        value=file_row_counts.get(hist_basename, 0),
    )
    tracker.add_metric(
        id="s6_today_rows_output",
        title="Today Rows with Neighbors",
        value=file_row_counts.get(today_basename, 0),
    )
    tracker.add_metric(
        id="s6_duration",
        title="Step 6 Duration",
        value=round(duration, 1),
    )

    print("Step 6 complete")
