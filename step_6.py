import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

import constants


def s6_load_index_series(log_transform=False):
    """Load index data and return a date -> index_value series."""
    if not os.path.exists(constants.S1_INDEX_FILE):
        print(
            f"Warning: {constants.S1_INDEX_FILE} not found. Index prices will be NaN."
        )
        return None
    idx_df = pd.read_csv(constants.S1_INDEX_FILE)
    idx_df["date"] = pd.to_datetime(idx_df["date"])
    if log_transform:
        idx_df["index_value"] = np.log(idx_df["index_value"].clip(lower=0.01))
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


def s6_load_sales_lookup(historical_file):
    """Load and prepare sales lookup data from a historical file."""
    sales_cols = ["gemrate_id", "grade", "date", "price"]

    df_sales_lookup = pd.read_parquet(historical_file, columns=sales_cols)
    df_sales_lookup["date"] = pd.to_datetime(
        df_sales_lookup["date"], format="mixed", errors="coerce"
    )
    df_sales_lookup = df_sales_lookup.dropna(subset=["date"])
    df_sales_lookup = df_sales_lookup.reset_index(drop=True)

    df_sales_lookup = df_sales_lookup.rename(
        columns={
            "gemrate_id": "neighbor_id",
            "date": "sale_date",
            "price": "sale_price",
        }
    )
    df_sales_lookup = df_sales_lookup.dropna()
    df_sales_lookup["neighbor_id"] = df_sales_lookup["neighbor_id"].astype(str)
    return df_sales_lookup


def run_step_6():
    print("Starting Step 6: Neighbor Features...")

    print("Loading neighbors...")
    df_neighbors = pd.read_parquet(constants.S5_OUTPUT_NEIGHBORS_FILE)
    df_neighbors = df_neighbors.rename(columns={"neighbors": "neighbor_id"})
    df_neighbors["neighbor_rank"] = df_neighbors.groupby("gemrate_id").cumcount() + 1

    files_to_process = [
        (
            constants.S3_HISTORICAL_DATA_FILE,
            "historical_data_with_neighbors.parquet",
            constants.S3_HISTORICAL_DATA_FILE,
            False,
        ),
        (
            constants.S3_TODAY_DATA_FILE,
            "today_data_with_neighbors.parquet",
            constants.S3_HISTORICAL_DATA_FILE,
            False,
        ),
        (
            constants.S3_HISTORICAL_DATA_LOG_FILE,
            "historical_data_log_with_neighbors.parquet",
            constants.S3_HISTORICAL_DATA_LOG_FILE,
            True,
        ),
        (
            constants.S3_TODAY_DATA_LOG_FILE,
            "today_data_log_with_neighbors.parquet",
            constants.S3_HISTORICAL_DATA_LOG_FILE,
            True,
        ),
    ]

    sales_lookup_cache = {}
    index_series_cache = {}

    for input_file, output_file, lookup_file, log_transform in files_to_process:
        print(f"\nProcessing {input_file}...")

        if not os.path.exists(input_file):
            print(f"  Skipping: {input_file} not found")
            continue

        if lookup_file not in sales_lookup_cache:
            print(f"  Loading sales lookup from {lookup_file}...")
            sales_lookup_cache[lookup_file] = s6_load_sales_lookup(lookup_file)
        df_sales_lookup = sales_lookup_cache[lookup_file]

        if log_transform not in index_series_cache:
            print(f"  Loading index data (log_transform={log_transform})...")
            index_series_cache[log_transform] = s6_load_index_series(
                log_transform=log_transform
            )
        index_series = index_series_cache[log_transform]

        try:
            parquet_file = pq.ParquetFile(input_file)
        except Exception as e:
            print(f"Could not open {input_file}: {e}")
            continue

        writer = None
        total_rows = 0
        batch_size = constants.S6_BATCH_SIZE

        for batch in tqdm(
            parquet_file.iter_batches(batch_size=batch_size),
            desc=f"Processing {os.path.basename(input_file)}",
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
                f"Adding neighbors to {input_file} complete. Saved {total_rows} rows to {output_file}."
            )
        else:
            print(f"No data processed for {input_file}")

    print("\nStep 6 complete")
