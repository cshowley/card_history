import os
import re
import sys

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

load_dotenv()


RUN_STEP_1_DOWNLOAD = False
RUN_STEP_2_FEATURE_PREP = False
RUN_STEP_3_TEXT_EMBEDDING = False
RUN_STEP_4_PRICE_EMBEDDING = False
RUN_STEP_5_NEIGHBOR_SEARCH = False
RUN_STEP_6_NEIGHBOR_PRICES = True


S1_MONGO_URL = os.getenv("MONGO_URL") or os.getenv("MONGO_URI")
S1_DB_NAME = "gemrate"
S1_EBAY_COLLECTION = "ebay_graded_items"
S1_PWCC_COLLECTION = "pwcc_graded_items"
S1_INDEX_API_URL = "https://price.collectorcrypt.com/api/indexes/modern"
S1_EBAY_MARKET_FILE = "market_ebay.csv"
S1_PWCC_MARKET_FILE = "market_pwcc.csv"
S1_INDEX_FILE = "index.csv"
S1_MONGO_MAX_TIME_MS = 6000000


S2_NUMBER_OF_BIDS_FILTER = 3
S2_N_SALES_BACK = 5
S2_WEEKS_BACK_LIST = [1, 2, 3, 4]
S2_FEATURES_PREPPED_FILE = "features_prepped.csv"
S2_INDEX_EMA_SHORT_SPAN = 12
S2_INDEX_EMA_LONG_SPAN = 26


S3_INPUT_CATALOG_FILE = "gemrate_pokemon_catalog_20260108.csv"
S3_OUTPUT_EMBEDDINGS_FILE = "text_embeddings.parquet"
S3_MODEL_NAME = "BAAI/bge-m3"


S4_WINDOW_SIZE = 4
S4_BATCH_SIZE = 1024
S4_EPOCHS = 100
S4_PRICE_EMBEDDING_DIM = 32
S4_LEARNING_RATE = 0.0001
S4_OUTPUT_PRICE_VECS_FILE = "price_embeddings.parquet"


S5_N_NEIGHBORS_PREPARE = 500
S5_OUTPUT_NEIGHBORS_FILE = "neighbors.parquet"


S6_N_NEIGHBORS = 5
S6_N_NEIGHBOR_SALES = 3
S6_BATCH_SIZE = 10000
S6_OUTPUT_FILE = "features_prepped_with_neighbors.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def s1_fetch_index_data(url):
    response = requests.get(url)
    response.raise_for_status()
    index_data = response.json()

    index_df = pd.DataFrame(index_data)
    index_df["date"] = pd.to_datetime(index_df["date"])
    index_df = index_df.rename(columns={"value": "index_value"})

    print(f"Downloaded {len(index_df)} index data points")
    print(f"Index date range: {index_df['date'].min()} to {index_df['date'].max()}")

    return index_df


def run_step_1():
    print("Starting Step 1: Download Sales...")
    if not S1_MONGO_URL:
        raise ValueError("MONGO_URL environment variable is not set")

    client = MongoClient(S1_MONGO_URL)
    db = client[S1_DB_NAME]

    print("Downloading eBay sales...")
    ebay_collection = db[S1_EBAY_COLLECTION]
    ebay_pipeline = [
        {
            "$match": {
                "$or": [
                    {"gemrate_data.universal_gemrate_id": {"$exists": True, "$ne": ""}},
                    {"gemrate_data.gemrate_id": {"$exists": True, "$ne": ""}},
                ],
                "item_data.date": {"$exists": True},
                "item_data.price": {"$exists": True},
            }
        },
        {
            "$project": {
                "gemrate_data.universal_gemrate_id": 1,
                "gemrate_data.gemrate_id": 1,
                "item_data.date": 1,
                "grading_company": 1,
                "gemrate_data.grade": 1,
                "item_data.price": 1,
                "item_data.number_of_bids": 1,
                "item_data.seller_name": 1,
                "_id": 1,
            }
        },
    ]
    ebay_results = ebay_collection.aggregate(
        ebay_pipeline, maxTimeMS=6000000, allowDiskUse=True
    )
    ebay_df = pd.json_normalize(list(ebay_results))
    print(f"  → eBay rows loaded: {len(ebay_df)}")
    ebay_df.to_csv(S1_EBAY_MARKET_FILE, index=False)

    print("Downloading PWCC/Fanatics sales...")
    pwcc_collection = db[S1_PWCC_COLLECTION]
    pwcc_pipeline = [
        {
            "$match": {
                "$or": [
                    {"gemrate_data.universal_gemrate_id": {"$exists": True, "$ne": ""}},
                    {"gemrate_data.gemrate_id": {"$exists": True, "$ne": ""}},
                ],
                "api_response.soldDate": {"$exists": True},
                "api_response.purchasePrice": {"$exists": True},
                "api_response.auctionType": {"$in": ["WEEKLY", "PREMIER"]},
            }
        },
        {
            "$project": {
                "gemrate_data.universal_gemrate_id": 1,
                "gemrate_data.gemrate_id": 1,
                "api_response.soldDate": 1,
                "api_response.purchasePrice": 1,
                "api_response.auctionType": 1,
                "api_response.gradingService": 1,
                "gemrate_data.grade": 1,
                "_id": 1,
            }
        },
    ]
    pwcc_results = pwcc_collection.aggregate(
        pwcc_pipeline, maxTimeMS=6000000, allowDiskUse=True
    )
    pwcc_df = pd.json_normalize(list(pwcc_results))
    print(f"  → PWCC rows loaded: {len(pwcc_df)}")
    pwcc_df.to_csv(S1_PWCC_MARKET_FILE, index=False)

    index_df = s1_fetch_index_data(S1_INDEX_API_URL)
    index_df.to_csv(S1_INDEX_FILE, index=False)
    print("Step 1 Complete.")


def s2_clean_grade(val):
    s = str(val).lower().strip().replace("g", "").replace("_", ".")
    if s in ["nan", "none", "", "0", "auth"]:
        return np.nan

    if "10b" in s or "10black" in s:
        return 11.0

    if any(x in s for x in ["pristine", "perfect", "10p"]):
        return 10.5

    match = re.search(r"(\d+(\.\d+)?)", s)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return np.nan
    return np.nan


def s2_process_grade(val):
    if pd.isna(val):
        return np.nan, np.nan
    floor_val = np.floor(val)
    half_val = 1.0 if (val - floor_val) > 0 else 0.0
    return floor_val, half_val


def s2_group_currencies(val):
    s = str(val).strip()
    if not s:
        return "Unknown"
    if s.startswith("$") or s[0].isdigit():
        return "$ (No Country Code)"
    return s


def s2_calculate_seller_popularity(df):
    print("Calculating seller popularity (expanding window)...")
    df = df.sort_values("date").copy()
    df["seller_cum_count"] = df.groupby("seller_name").cumcount() + 1
    df["global_cum_count"] = np.arange(1, len(df) + 1)
    df["seller_popularity"] = df["seller_cum_count"] / df["global_cum_count"]
    df = df.drop(columns=["seller_cum_count", "global_cum_count"])
    return df


def s2_create_previous_sale_features(df, n_sales_back):
    print(f"Generating {n_sales_back} lag features...")
    index_series = None
    idx_df = pd.read_csv(S1_INDEX_FILE)
    idx_df["date"] = pd.to_datetime(idx_df["date"])
    idx_df = idx_df.groupby("date")["index_value"].mean()
    index_series = idx_df
    df = df.sort_values(["gemrate_id", "grade", "date"]).reset_index(drop=True)

    feature_cols = [
        "price",
        "half_grade",
        "grade_co_BGS",
        "grade_co_CGC",
        "grade_co_PSA",
    ]
    # Filter to only columns that exist in the dataframe
    feature_cols = [col for col in feature_cols if col in df.columns]
    new_columns = []

    for n in range(1, n_sales_back + 1):
        prefix = f"prev_{n}"
        for col in feature_cols:
            new_col = f"{prefix}_{col}"
            df[new_col] = df.groupby(["gemrate_id", "grade"])[col].shift(n)
            new_columns.append(new_col)

        days_col = f"{prefix}_days_ago"
        prev_date = df.groupby(["gemrate_id", "grade"])["date"].shift(n)
        df[days_col] = (df["date"] - prev_date).dt.days
        new_columns.append(days_col)

        idx_col = f"{prefix}_index_price"
        if index_series is not None:
            df[idx_col] = prev_date.map(index_series)
        else:
            df[idx_col] = np.nan
        new_columns.append(idx_col)

    return df, new_columns


def s2_create_lookback_features(df, weeks_back_list):
    print("Generating lookback features...")
    df = df.sort_values(["gemrate_id", "grade", "date"]).reset_index(drop=True)
    df["week_start"] = df["date"].dt.to_period("W").dt.start_time

    agg_cols = [
        "price",
        "half_grade",
        "seller_popularity",
        "grade_co_BGS",
        "grade_co_CGC",
        "grade_co_PSA",
    ]
    # Filter to only columns that exist in the dataframe
    agg_cols = [col for col in agg_cols if col in df.columns]

    weekly_agg = (
        df.groupby(["gemrate_id", "grade", "week_start"])[agg_cols].mean().reset_index()
    )

    base_rename = {c: f"avg_{c}" for c in agg_cols}
    weekly_agg = weekly_agg.rename(columns=base_rename)

    result_df = df.copy()
    new_columns = []

    for w in weeks_back_list:
        shifted = weekly_agg.copy()
        shifted["join_week"] = shifted["week_start"] + pd.Timedelta(weeks=w)

        suffix = f"_{w}w_ago"
        cols_to_add = list(base_rename.values())
        rename_dict = {c: f"{c}{suffix}" for c in cols_to_add}
        shifted = shifted.rename(columns=rename_dict)

        final_cols = list(rename_dict.values())
        new_columns.extend(final_cols)

        result_df = result_df.merge(
            shifted[["gemrate_id", "grade", "join_week"] + final_cols],
            left_on=["gemrate_id", "grade", "week_start"],
            right_on=["gemrate_id", "grade", "join_week"],
            how="left",
        ).drop(columns=["join_week"])

    return result_df, new_columns


def s2_create_adjacent_grade_features(df, n_sales_back):
    print("Generating adjacent grade features...")
    df = df.sort_values(["gemrate_id", "grade", "date"]).reset_index(drop=True)

    feature_cols = [
        "price",
        "half_grade",
        "grade_co_BGS",
        "grade_co_CGC",
        "grade_co_PSA",
        "seller_popularity",
    ]
    # Filter to only columns that exist in the dataframe
    feature_cols = [col for col in feature_cols if col in df.columns]
    new_columns = []

    ref = df[["gemrate_id", "grade", "date"] + feature_cols].copy()
    ref["_sale_idx"] = ref.groupby(["gemrate_id", "grade"]).cumcount()

    for direction, grade_offset in [("above", 1), ("below", -1)]:
        target_grade = (df["grade"] + grade_offset).clip(lower=1.0, upper=10.0)

        df_lookup = df[["gemrate_id", "date"]].copy()
        df_lookup["target_grade"] = target_grade
        df_lookup["_orig_row"] = df.index
        df_lookup = df_lookup.sort_values("date")

        ref_lookup = ref.rename(
            columns={"grade": "target_grade", "date": "ref_date"}
        ).sort_values("ref_date")

        merged = pd.merge_asof(
            df_lookup,
            ref_lookup[["gemrate_id", "target_grade", "ref_date", "_sale_idx"]],
            left_on="date",
            right_on="ref_date",
            by=["gemrate_id", "target_grade"],
            direction="backward",
            allow_exact_matches=False,
        )

        for n in range(1, n_sales_back + 1):
            suffix = f"prev_{n}_{direction}"
            merged["_lookup_idx"] = merged["_sale_idx"] - (n - 1)

            ref_feats = ref[
                ["gemrate_id", "grade", "_sale_idx", "date"] + feature_cols
            ].copy()
            ref_feats.columns = [
                "gemrate_id",
                "target_grade",
                "_lookup_idx",
                "prev_date",
            ] + feature_cols

            with_feats = merged.merge(
                ref_feats,
                on=["gemrate_id", "target_grade", "_lookup_idx"],
                how="left",
            )
            with_feats = with_feats.set_index("_orig_row").reindex(df.index)

            for col in feature_cols:
                new_col = f"{suffix}_{col}"
                df[new_col] = with_feats[col]
                new_columns.append(new_col)

            df[f"{suffix}_days_ago"] = (
                with_feats["date"] - with_feats["prev_date"]
            ).dt.days
            new_columns.append(f"{suffix}_days_ago")

    return df, new_columns


def s2_load_and_join_index_data(df, filepath):
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found. Skipping index data.")
        return df

    print(f"Joining index data from {filepath}...")
    idx_df = pd.read_csv(filepath)
    idx_df["date"] = pd.to_datetime(idx_df["date"])
    idx_df = idx_df.sort_values("date")

    idx_df["index_change_1d"] = idx_df["index_value"].diff()
    idx_df["index_change_1w"] = idx_df["index_value"].diff(7)
    idx_df["index_ema_12"] = (
        idx_df["index_value"].ewm(span=S2_INDEX_EMA_SHORT_SPAN).mean()
    )
    idx_df["index_ema_26"] = (
        idx_df["index_value"].ewm(span=S2_INDEX_EMA_LONG_SPAN).mean()
    )

    df = df.merge(idx_df, on="date", how="left")
    return df


def s2_load_and_clean_ebay(filepath):
    """Load and clean eBay data."""
    print(f"Loading eBay data from {filepath}...")
    df = pd.read_csv(filepath)

    df["date"] = pd.to_datetime(df["item_data.date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["universal_gemrate_id"] = df["gemrate_data.universal_gemrate_id"]
    df["gemrate_id"] = df["gemrate_data.gemrate_id"]
    df.loc[df["universal_gemrate_id"].notna(), "gemrate_id"] = df.loc[
        df["universal_gemrate_id"].notna(), "universal_gemrate_id"
    ]
    df = df.dropna(subset=["gemrate_id"])
    df["grade"] = df["gemrate_data.grade"].apply(s2_clean_grade)
    df = df.dropna(subset=["grade"])
    df[["grade", "half_grade"]] = df["grade"].apply(
        lambda x: pd.Series(s2_process_grade(x))
    )

    currency_groups = (
        df["item_data.price"].astype(str).str.split().str[0].apply(s2_group_currencies)
    )
    df = df.loc[currency_groups.isin(["$ (No Country Code)", "US"])]
    df["price"] = np.log(
        df["item_data.price"]
        .astype(str)
        .str.replace(r"\D+", "", regex=True)
        .astype(float)
        .clip(lower=0.01)
    )

    df["number_of_bids"] = df["item_data.number_of_bids"]
    df = df[df["number_of_bids"] >= S2_NUMBER_OF_BIDS_FILTER]

    df["grading_company"] = df["grading_company"].fillna("Unknown")
    df["source"] = "ebay"

    print(f"  → eBay cleaned: {len(df)} rows")
    return df


def s2_load_and_clean_pwcc(filepath):
    """Load and clean PWCC/Fanatics data."""
    print(f"Loading PWCC data from {filepath}...")
    if not os.path.exists(filepath):
        print(f"  → {filepath} not found, skipping PWCC")
        return pd.DataFrame()

    df = pd.read_csv(filepath)
    if df.empty:
        return df

    df["api_response.soldDate"] = df["api_response.soldDate"].replace({None: pd.NaT})
    df = df.dropna(subset=["api_response.soldDate"])
    df["api_response.soldDate"] = (
        df["api_response.soldDate"]
        .astype(str)
        .str.replace(r" [A-Z]{3,4}$", "", regex=True)
    )
    df["date"] = pd.to_datetime(df["api_response.soldDate"], errors="coerce")
    df = df.dropna(subset=["date"])

    df["universal_gemrate_id"] = df["gemrate_data.universal_gemrate_id"]
    df["gemrate_id"] = df["gemrate_data.gemrate_id"]
    df.loc[df["universal_gemrate_id"].notna(), "gemrate_id"] = df.loc[
        df["universal_gemrate_id"].notna(), "universal_gemrate_id"
    ]
    df = df.dropna(subset=["gemrate_id"])

    df["grade"] = df["grade"].apply(s2_clean_grade)
    df = df.dropna(subset=["grade"])
    df[["grade", "half_grade"]] = df["grade"].apply(
        lambda x: pd.Series(s2_process_grade(x))
    )

    df["price"] = pd.to_numeric(df["api_response.purchasePrice"], errors="coerce")
    df = df.dropna(subset=["price"])
    df["price"] = np.log(df["price"].clip(lower=0.01))

    df["grading_company"] = df["api_response.gradingService"].fillna("Unknown")
    df["seller_name"] = "fanatics"
    conditions = [
        df["api_response.auctionType"] == "WEEKLY",
        df["api_response.auctionType"] == "PREMIER",
    ]
    choices = ["fanatics_weekly", "fanatics_premier"]
    df["source"] = np.select(conditions, choices, default="fanatics_unknown")

    print(f"  → PWCC cleaned: {len(df)} rows")
    return df


def run_step_2():
    print("Starting Step 2: Feature Prep...")

    ebay_df = s2_load_and_clean_ebay(S1_EBAY_MARKET_FILE)
    pwcc_df = s2_load_and_clean_pwcc(S1_PWCC_MARKET_FILE)

    common_cols = [
        "gemrate_id",
        "date",
        "price",
        "grade",
        "half_grade",
        "grading_company",
        "seller_name",
        "source",
    ]

    ebay_subset = ebay_df[[c for c in common_cols if c in ebay_df.columns]].copy()
    pwcc_subset = pwcc_df[[c for c in common_cols if c in pwcc_df.columns]].copy()

    print("Merging eBay and PWCC data...")
    df = pd.concat([ebay_subset, pwcc_subset], ignore_index=True)
    print(f"  → Total merged: {len(df)} rows")
    print(f"  → Source distribution: {df['source'].value_counts().to_dict()}")

    df["source_ebay"] = (df["source"] == "ebay").astype(int)
    df["source_fanatics_weekly"] = (df["source"] == "fanatics_weekly").astype(int)
    df["source_fanatics_premier"] = (df["source"] == "fanatics_premier").astype(int)

    dummies = pd.get_dummies(df["grading_company"], prefix="grade_co", dtype=int)
    df = pd.concat([df, dummies], axis=1)

    df = s2_calculate_seller_popularity(df)

    columns_to_keep = [
        "gemrate_id",
        "date",
        "price",
        "grade",
        "half_grade",
        "source_ebay",
        "source_fanatics_weekly",
        "source_fanatics_premier",
        "seller_popularity",
    ]
    columns_to_keep.extend(dummies.columns.tolist())
    df = df[columns_to_keep].copy()

    catalog_df = pd.read_csv(S3_INPUT_CATALOG_FILE, low_memory=False)
    mask_univ = catalog_df["UNIVERSAL_GEMRATE_ID"].notna()
    catalog_df.loc[mask_univ, "GEMRATE_ID"] = catalog_df.loc[
        mask_univ, "UNIVERSAL_GEMRATE_ID"
    ]
    catalog_df = catalog_df.drop_duplicates(subset=["GEMRATE_ID"])

    valid_ids = set(catalog_df["GEMRATE_ID"].astype(str))
    initial_count = len(df)

    df = df[df["gemrate_id"].astype(str).isin(valid_ids)]
    print(
        f"  → Filtered {initial_count - len(df)} rows not in catalog. Remaining: {len(df)}"
    )

    df, prev_cols = s2_create_previous_sale_features(df, S2_N_SALES_BACK)
    df, lookback_cols = s2_create_lookback_features(df, S2_WEEKS_BACK_LIST)
    df, adj_cols = s2_create_adjacent_grade_features(df, S2_N_SALES_BACK)
    df = s2_load_and_join_index_data(df, S1_INDEX_FILE)

    df = df.drop("week_start", axis=1)
    print(f"Saving {len(df)} rows to {S2_FEATURES_PREPPED_FILE}...")
    df.to_csv(S2_FEATURES_PREPPED_FILE, index=False)
    print("Step 2 Complete.")


def run_step_3():
    print("Starting Step 3: Text Embedding...")
    if not os.path.exists(S3_INPUT_CATALOG_FILE):
        print(f"Error: Input file '{S3_INPUT_CATALOG_FILE}' not found.")
        sys.exit(1)

    print(f"Loading data from {S3_INPUT_CATALOG_FILE}...")
    df = pd.read_csv(S3_INPUT_CATALOG_FILE, low_memory=False)

    print(f"Initial rows: {len(df)}")

    mask_univ = df["UNIVERSAL_GEMRATE_ID"].notna()
    df.loc[mask_univ, "GEMRATE_ID"] = df.loc[mask_univ, "UNIVERSAL_GEMRATE_ID"]
    df = df.drop_duplicates(subset=["GEMRATE_ID"])

    if "YEAR" in df.columns:
        df["YEAR"] = df["YEAR"].astype(str).str.split("-").str[0].str.strip()

    required_cols = [
        "GEMRATE_ID",
        "YEAR",
        "SET_NAME",
        "NAME",
        "CARD_NUMBER",
        "PARALLEL",
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(
            f"Warning: Missing columns {missing_cols}. Proceeding with available columns..."
        )

    cols_to_use = [c for c in required_cols if c in df.columns]
    df = df[cols_to_use].copy()

    df = df.fillna("")
    df = df.astype(str)
    df = df.reset_index(drop=True)

    print(f" Done. Rows after cleaning: {len(df)}")

    print("Creating text for embedding...")

    df["embedding_text"] = ""
    for col in ["YEAR", "SET_NAME", "NAME", "CARD_NUMBER", "PARALLEL"]:
        if col in df.columns:
            df["embedding_text"] += df[col] + " "

    df["embedding_text"] = (
        df["embedding_text"].str.replace(r"\s+", " ", regex=True).str.strip()
    )

    print(f"Loading model: {S3_MODEL_NAME}...")
    model = SentenceTransformer(S3_MODEL_NAME, device=DEVICE)

    print("Generating embeddings...")

    embeddings = model.encode(
        df["embedding_text"].tolist(),
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    df["embedding_vector"] = list(embeddings)
    df["gemrate_id"] = df["GEMRATE_ID"]
    output_df = df[["gemrate_id", "embedding_vector"]]

    print(f"Saving {len(output_df)} rows to {S3_OUTPUT_EMBEDDINGS_FILE}...")
    output_df.to_parquet(S3_OUTPUT_EMBEDDINGS_FILE, index=False)
    print("Step 3 Complete.")


class LSTMAutoencoder(nn.Module):
    def __init__(
        self, seq_len=S4_WINDOW_SIZE, n_features=1, embedding_dim=S4_PRICE_EMBEDDING_DIM
    ):
        super(LSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim

        self.encoder_lstm = nn.LSTM(
            input_size=n_features, hidden_size=embedding_dim, batch_first=True
        )

        self.decoder_lstm = nn.LSTM(
            input_size=embedding_dim, hidden_size=embedding_dim, batch_first=True
        )

        self.output_layer = nn.Linear(embedding_dim, n_features)

    def forward(self, x):
        _, (h_n, _) = self.encoder_lstm(x)
        embedding = h_n.squeeze(0)

        repeat_embedding = embedding.unsqueeze(1).repeat(1, self.seq_len, 1)

        decoder_output, _ = self.decoder_lstm(repeat_embedding)

        x_recon = self.output_layer(decoder_output)

        return x_recon, embedding


def s4_normalize_grades(df):
    print("Normalizing grades...")
    grade_counts = df.groupby(["gemrate_id", "grade"]).size().reset_index(name="count")
    base_grades = (
        grade_counts.sort_values(["gemrate_id", "count"], ascending=[True, False])
        .drop_duplicates("gemrate_id")
        .set_index("gemrate_id")["grade"]
    )

    avg_prices = df.groupby(["gemrate_id", "grade"])["price"].mean()

    multipliers = avg_prices.reset_index(name="avg_price")
    multipliers["base_grade"] = multipliers["gemrate_id"].map(base_grades)

    base_prices = avg_prices.reset_index()
    base_prices = base_prices[
        base_prices["grade"] == base_prices["gemrate_id"].map(base_grades)
    ]
    base_prices = base_prices.set_index("gemrate_id")["price"]

    multipliers["base_price"] = multipliers["gemrate_id"].map(base_prices)
    multipliers["multiplier"] = multipliers["avg_price"] / multipliers["base_price"]

    df = df.merge(
        multipliers[["gemrate_id", "grade", "multiplier"]],
        on=["gemrate_id", "grade"],
        how="left",
    )

    df["normalized_price"] = df["price"] / df["multiplier"]

    return df


def s4_prepare_price_matrix(df):
    print("Preparing price matrix...")
    df["date"] = pd.to_datetime(df["date"], format="mixed")

    pivot_df = df.groupby(["date", "gemrate_id"])["normalized_price"].mean().unstack()

    pivot_df = pivot_df.resample("W").mean()
    pivot_df = pivot_df.ffill(limit=8)
    print("ids before drop", df["gemrate_id"].nunique())
    pivot_df = pivot_df.dropna(axis=1)
    print(f"Price Matrix Shape after cleaning: {pivot_df.shape}")

    if pivot_df.empty:
        print("Warning: Price matrix is empty after dropna/ffill.")
        return pivot_df, [], []

    log_returns = np.log(pivot_df / pivot_df.shift(1)).dropna()

    data_values = log_returns.values
    card_ids = log_returns.columns
    print("ids after drop", len(set(card_ids)))
    if data_values.shape[0] < S4_WINDOW_SIZE:
        print("Not enough time points for windowing.")
        return pivot_df, [], []

    print("Generating sliding windows...")

    raw_windows = []
    for i in range(len(data_values) - S4_WINDOW_SIZE + 1):
        window = data_values[i : i + S4_WINDOW_SIZE, :]
        raw_windows.append(window)

    raw_windows = np.array(raw_windows)

    processed = np.transpose(raw_windows, (2, 0, 1))

    processed = processed.reshape(-1, S4_WINDOW_SIZE)

    means = processed.mean(axis=1, keepdims=True)
    stds = processed.std(axis=1, keepdims=True) + 1e-8
    processed = (processed - means) / stds

    num_time_steps = raw_windows.shape[0]
    ids_map = np.repeat(card_ids.values, num_time_steps)

    X = processed[:, :, np.newaxis]

    return pivot_df, X, ids_map


def s4_train_and_extract(X_data, ids_map):
    print("Starting Training Process...")

    indices = np.arange(len(X_data))
    np.random.shuffle(indices)
    split_idx = int(len(X_data) * 0.8)
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]

    X_train = X_data[train_idx]
    X_val = X_data[val_idx]

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=S4_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=S4_BATCH_SIZE, shuffle=False)

    print("Phase 1: Finding optimal epochs (Holdout Validation)...")
    model = LSTMAutoencoder().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=S4_LEARNING_RATE)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(S4_EPOCHS):
        model.train()
        for (batch_x,) in train_loader:
            batch_x = batch_x.to(DEVICE)
            optimizer.zero_grad()
            recon, _ = model(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for (batch_x,) in val_loader:
                batch_x = batch_x.to(DEVICE)
                recon, _ = model(batch_x)
                loss = criterion(recon, batch_x)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}/{S4_EPOCHS} - Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1

    print(
        f"Optimal number of epochs determined: {best_epoch} (Loss: {best_val_loss:.6f})"
    )

    print(f"Phase 2: Retraining on full dataset for {best_epoch} epochs...")
    full_dataset = TensorDataset(torch.tensor(X_data, dtype=torch.float32))
    full_train_loader = DataLoader(full_dataset, batch_size=S4_BATCH_SIZE, shuffle=True)

    final_model = LSTMAutoencoder().to(DEVICE)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=S4_LEARNING_RATE)

    final_model.train()
    for epoch in range(best_epoch):
        total_loss = 0
        for (batch_x,) in full_train_loader:
            batch_x = batch_x.to(DEVICE)
            optimizer.zero_grad()
            recon, _ = final_model(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(
            f"Full Train Epoch {epoch + 1}/{best_epoch}, Loss: {total_loss / len(full_train_loader):.6f}"
        )

    print("Extracting price vectors...")
    final_model.eval()

    extraction_loader = DataLoader(
        full_dataset, batch_size=S4_BATCH_SIZE * 2, shuffle=False
    )

    all_embeddings = []
    with torch.no_grad():
        for (batch_x,) in extraction_loader:
            batch_x = batch_x.to(DEVICE)
            _, emb = final_model(batch_x)
            all_embeddings.append(emb.cpu().numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)

    df_vecs = pd.DataFrame(all_embeddings)
    df_vecs["gemrate_id"] = ids_map

    mean_vecs = df_vecs.groupby("gemrate_id").mean()

    return mean_vecs


def run_step_4():
    print("Starting Step 4: Price Embedding...")
    if not os.path.exists(S2_FEATURES_PREPPED_FILE):
        print(f"File {S2_FEATURES_PREPPED_FILE} not found.")
        return

    df_sales = pd.read_csv(
        S2_FEATURES_PREPPED_FILE,
        usecols=["gemrate_id", "grade", "date", "price"],
        dtype={"gemrate_id": str},
    )

    df_norm = s4_normalize_grades(df_sales)

    _, X_windows, ids_map = s4_prepare_price_matrix(df_norm)

    if len(X_windows) == 0:
        print("No windows generated. Exiting.")
        return

    price_vecs = s4_train_and_extract(X_windows, ids_map)
    print("Price Vectors Head:")
    print(price_vecs.head())

    print(f"Saving price vectors to {S4_OUTPUT_PRICE_VECS_FILE}...")
    price_vecs.to_parquet(S4_OUTPUT_PRICE_VECS_FILE)
    print("Step 4 Complete.")


def s5_build_search_matrices(price_vecs, embedding_file):
    print("Loading Text Embeddings...")
    text_emb_df = pd.read_parquet(embedding_file)

    print(f"Stacking text embeddings for {len(text_emb_df)} items...")

    if "gemrate_id" in text_emb_df.columns:
        text_emb_df = text_emb_df.set_index("gemrate_id")

    matrix = np.array(text_emb_df["embedding_vector"].tolist())

    norm = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    matrix_norm = matrix / norm

    text_embeddings_norm = pd.DataFrame(matrix_norm, index=text_emb_df.index)

    print("Preparing Price Embeddings...")

    p_matrix = price_vecs.values
    p_norm = np.linalg.norm(p_matrix, axis=1, keepdims=True) + 1e-10
    p_matrix_norm = p_matrix / p_norm

    price_embeddings_norm = pd.DataFrame(p_matrix_norm, index=price_vecs.index)

    common_ids = text_embeddings_norm.index.intersection(price_embeddings_norm.index)
    common_ids = common_ids.sort_values()

    print(
        f"Building Search Database with {len(common_ids)} items (intersection of Text & Price)..."
    )

    db_text_np = text_embeddings_norm.loc[common_ids].values
    db_price_np = price_embeddings_norm.loc[common_ids].values

    db_text_tensor = torch.tensor(db_text_np, dtype=torch.float32, device=DEVICE)
    db_price_tensor = torch.tensor(db_price_np, dtype=torch.float32, device=DEVICE)

    print("Searcher ready.")
    return (
        db_text_tensor,
        db_price_tensor,
        common_ids,
        text_embeddings_norm,
        price_embeddings_norm,
    )


def s5_prepare_matrices(price_vecs, embedding_file):
    print("Loading Text Embeddings...")
    text_emb_df = pd.read_parquet(embedding_file)
    if "gemrate_id" in text_emb_df.columns:
        text_emb_df = text_emb_df.set_index("gemrate_id")

    t_matrix = np.array(text_emb_df["embedding_vector"].tolist())
    t_norm = np.linalg.norm(t_matrix, axis=1, keepdims=True) + 1e-10
    text_emb_df_norm = pd.DataFrame(t_matrix / t_norm, index=text_emb_df.index)

    print("Preparing Price Embeddings...")

    p_matrix = price_vecs.values
    p_norm = np.linalg.norm(p_matrix, axis=1, keepdims=True) + 1e-10
    price_emb_df_norm = pd.DataFrame(p_matrix / p_norm, index=price_vecs.index)

    db_ids = price_emb_df_norm.index.intersection(text_emb_df_norm.index).sort_values()

    print(
        f"Database (Potential Neighbors): {len(db_ids)} items (Must have Text + Price)"
    )

    db_text_tensor = torch.tensor(
        text_emb_df_norm.loc[db_ids].values, dtype=torch.float32, device=DEVICE
    )
    db_price_tensor = torch.tensor(
        price_emb_df_norm.loc[db_ids].values, dtype=torch.float32, device=DEVICE
    )

    query_ids = text_emb_df_norm.index
    print(f"Query Set (Items needing neighbors): {len(query_ids)} items")

    return (
        db_text_tensor,
        db_price_tensor,
        db_ids,
        text_emb_df_norm,
        price_emb_df_norm,
        query_ids,
    )


def run_step_5():
    print("Starting Step 5: Neighbor Search (Robust)...")

    if not os.path.exists(S3_OUTPUT_EMBEDDINGS_FILE):
        print(f"Embeddings file {S3_OUTPUT_EMBEDDINGS_FILE} not found.")
        return

    if not os.path.exists(S4_OUTPUT_PRICE_VECS_FILE):
        print(f"Price vectors file {S4_OUTPUT_PRICE_VECS_FILE} not found.")
        return

    price_vecs = pd.read_parquet(S4_OUTPUT_PRICE_VECS_FILE)
    if "gemrate_id" in price_vecs.columns:
        price_vecs = price_vecs.set_index("gemrate_id")

    (
        db_text_tensor,
        db_price_tensor,
        db_ids,
        all_text_norm,
        all_price_norm,
        query_ids,
    ) = s5_prepare_matrices(price_vecs, S3_OUTPUT_EMBEDDINGS_FILE)

    q_text_np = all_text_norm.loc[query_ids].values
    q_text = torch.tensor(q_text_np, dtype=torch.float32, device=DEVICE)

    q_price_aligned = all_price_norm.reindex(query_ids, fill_value=0.0)
    q_price = torch.tensor(q_price_aligned.values, dtype=torch.float32, device=DEVICE)

    print("Computing similarity matrix...")
    batch_size = 4096

    all_gemrate_ids = []
    all_neighbors = []
    all_scores = []

    num_queries = len(query_ids)

    for i in tqdm(range(0, num_queries, batch_size), desc="Searching"):
        end_idx = min(i + batch_size, num_queries)

        batch_q_text = q_text[i:end_idx]
        batch_q_price = q_price[i:end_idx]

        with torch.no_grad():
            sim_text = torch.matmul(batch_q_text, db_text_tensor.T)

            sim_price = torch.matmul(batch_q_price, db_price_tensor.T)

            total_sim = sim_text + sim_price

            k_val = min(S5_N_NEIGHBORS_PREPARE + 1, total_sim.shape[1])
            top_vals, top_idxs = torch.topk(total_sim, k=k_val, dim=1)

            top_idxs = top_idxs.cpu().numpy()
            top_vals = top_vals.cpu().numpy()

        batch_ids = query_ids[i:end_idx]

        for local_idx, query_id in enumerate(batch_ids):
            indices = top_idxs[local_idx]
            vals = top_vals[local_idx]

            neighbor_ids = db_ids[indices]

            mask = neighbor_ids != query_id
            final_ids = neighbor_ids[mask][:S5_N_NEIGHBORS_PREPARE]
            final_scores = vals[mask][:S5_N_NEIGHBORS_PREPARE]

            count = len(final_ids)
            if count > 0:
                all_gemrate_ids.extend([query_id] * count)
                all_neighbors.extend(final_ids)
                all_scores.extend(final_scores)

    results_df = pd.DataFrame(
        {"gemrate_id": all_gemrate_ids, "neighbors": all_neighbors, "score": all_scores}
    )

    print(
        f"Saving final output with {len(results_df)} rows to {S5_OUTPUT_NEIGHBORS_FILE}..."
    )
    print(
        f"Unique Query IDs processed: {results_df['gemrate_id'].nunique()} / {num_queries}"
    )
    results_df.to_parquet(S5_OUTPUT_NEIGHBORS_FILE)
    print("Step 5 Complete.")


def s6_load_index_series():
    """Load index data and return a date -> index_value series."""
    if not os.path.exists(S1_INDEX_FILE):
        print(f"Warning: {S1_INDEX_FILE} not found. Index prices will be NaN.")
        return None
    idx_df = pd.read_csv(S1_INDEX_FILE)
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

    merged = merged[merged["dense_neighbor_rank"] <= S6_N_NEIGHBORS]
    merged["sale_rank"] = (
        merged.groupby(["_row_id", "dense_neighbor_rank"]).cumcount() + 1
    )
    merged = merged[merged["sale_rank"] <= S6_N_NEIGHBOR_SALES]

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

    for n in range(1, S6_N_NEIGHBORS + 1):
        col_score = f"neighbor_{n}_score"
        feature_data[col_score] = pivot_score[n] if n in pivot_score.columns else np.nan

        for s in range(1, S6_N_NEIGHBOR_SALES + 1):
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
    chunk_result["_row_id"] = chunk_result.index
    return chunk_result


def run_step_6():
    print("Starting Step 6: Neighbor Features...")
    print("Loading sales data...")
    df_sales = pd.read_csv(S2_FEATURES_PREPPED_FILE, low_memory=False)
    df_sales["date"] = pd.to_datetime(df_sales["date"], format="mixed", errors="coerce")
    df_sales = df_sales.dropna(subset=["date"])
    df_sales = df_sales.reset_index(drop=True)
    df_sales["_row_id"] = df_sales.index

    print("Loading neighbors...")
    df_neighbors = pd.read_parquet(S5_OUTPUT_NEIGHBORS_FILE)
    df_neighbors = df_neighbors.rename(columns={"neighbors": "neighbor_id"})

    df_neighbors["neighbor_rank"] = df_neighbors.groupby("gemrate_id").cumcount() + 1

    print("Loading index data...")
    index_series = s6_load_index_series()

    print("Pre-preparing neighbor sales lookup...")
    neighbor_sales = df_sales[["gemrate_id", "grade", "date", "price"]].copy()
    neighbor_sales = neighbor_sales.rename(
        columns={
            "gemrate_id": "neighbor_id",
            "date": "sale_date",
            "price": "sale_price",
        }
    )
    neighbor_sales = neighbor_sales.dropna()

    neighbor_sales["neighbor_id"] = neighbor_sales["neighbor_id"].astype(str)

    total_rows = len(df_sales)
    results = []

    print(f"Processing {total_rows} rows in batches of {S6_BATCH_SIZE}...")

    for start_idx in tqdm(range(0, total_rows, S6_BATCH_SIZE)):
        end_idx = min(start_idx + S6_BATCH_SIZE, total_rows)
        chunk_df = df_sales.iloc[start_idx:end_idx].copy()

        chunk_features = s6_process_chunk(
            chunk_df, df_neighbors, neighbor_sales, index_series
        )
        results.append(chunk_features)

    print("Concatenating batches...")
    all_features = pd.concat(results, ignore_index=True)

    print("Final merge...")
    final_df = df_sales.merge(all_features, on="_row_id", how="left")
    final_df.drop(columns=["_row_id"], inplace=True)

    print(f"Saving {len(final_df)} rows to {S6_OUTPUT_FILE}...")
    final_df.to_csv(S6_OUTPUT_FILE, index=False)
    print("Step 6 Complete.")


if __name__ == "__main__":
    if RUN_STEP_1_DOWNLOAD:
        run_step_1()

    if RUN_STEP_2_FEATURE_PREP:
        run_step_2()

    if RUN_STEP_3_TEXT_EMBEDDING:
        run_step_3()

    if RUN_STEP_4_PRICE_EMBEDDING:
        run_step_4()

    if RUN_STEP_5_NEIGHBOR_SEARCH:
        run_step_5()

    if RUN_STEP_6_NEIGHBOR_PRICES:
        run_step_6()
