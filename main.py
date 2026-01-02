import os
import re
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import FeatureHasher
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from tqdm import tqdm

load_dotenv()

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Pipeline Switches
RUN_STEP_1_DOWNLOAD = False
RUN_STEP_2_FEATURE_PREP = False
RUN_STEP_3_TRAIN_EMBEDDING = False
RUN_STEP_4_MERGE_PREDICTIONS = False

# Part 1: Download Sales Config
S1_MONGO_URL = os.getenv("MONGO_URL") or os.getenv("MONGO_URI")
S1_DB_NAME = "gemrate"
S1_EBAY_COLLECTION = "ebay_graded_items"
S1_PWCC_COLLECTION = "pwcc_graded_items"
S1_INDEX_API_URL = "https://price.collectorcrypt.com/api/indexes/modern"
S1_EBAY_MARKET_FILE = "market_ebay.csv"
S1_PWCC_MARKET_FILE = "market_pwcc.csv"
S1_INDEX_FILE = "index.csv"

# Part 2: Feature Prep Config
S2_NUMBER_OF_BIDS_FILTER = 3
S2_N_SALES_BACK = 5
S2_WEEKS_BACK_LIST = [1, 2, 3, 4]
S2_FEATURES_PREPPED_FILE = "features_prepped.csv"

# Part 3: Train Embedding Config
S3_RANDOM_SEED = 42
S3_HASH_N_FEATURES = 2**12
S3_SVD_N_COMPONENTS = 256
S3_OHE_MIN_FREQUENCY = 10
S3_VOLUME_CAP_PER_WEEK = 200.0
S3_ENCODER_HIDDEN_1 = 2048
S3_ENCODER_HIDDEN_2 = 1024
S3_ENCODER_OUT_DIM = 768
S3_GRADE_EMBED_DIM = 64
S3_DROPOUT_P = 0.1
S3_FINETUNE_EPOCHS = 20
S3_FINETUNE_BATCH_SIZE = 512
S3_FINETUNE_LR = 5e-4
S3_FINETUNE_TAU = 0.07
S3_FINETUNE_VAL_SPLIT = 0.1
S3_FINETUNE_PATIENCE = 5
S3_EMBED_BATCH_SIZE = 4096
S3_CARD_COLLECTION = "gemrate_pokemon_cards"
S3_EMBEDDING_KEYS_FILE = "embedding_keys.csv"
S3_EMBEDDING_VECTORS_FILE = "card_vectors_768.npy"

# Part 4: Merge Predictions Config
S4_OUTPUT_FILE = "features_prepped_with_neighbors.csv"
S4_N_NEIGHBORS = 5
S4_LOOKBACK_DAYS = 30
S4_K_CANDIDATES = 50

# ==============================================================================
# PART 1: DOWNLOAD SALES FUNCTIONS
# ==============================================================================


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

    # Download eBay data
    print("Downloading eBay sales...")
    ebay_collection = db[S1_EBAY_COLLECTION]
    ebay_pipeline = [
        {
            "$match": {
                "gemrate_data.universal_gemrate_id": {"$exists": True, "$ne": ""},
                "item_data.date": {"$exists": True},
                "item_data.price": {"$exists": True},
            }
        },
        {
            "$project": {
                "gemrate_data.universal_gemrate_id": 1,
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

    # Download PWCC/Fanatics data
    print("Downloading PWCC/Fanatics sales...")
    pwcc_collection = db[S1_PWCC_COLLECTION]
    pwcc_pipeline = [
        {
            "$match": {
                "gemrate_data.universal_gemrate_id": {"$exists": True, "$ne": ""},
                "api_response.soldDate": {"$exists": True},
                "api_response.purchasePrice": {"$exists": True},
            }
        },
        {
            "$project": {
                "gemrate_data.universal_gemrate_id": 1,
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

    # Download index data
    index_df = s1_fetch_index_data(S1_INDEX_API_URL)
    index_df.to_csv(S1_INDEX_FILE, index=False)
    print("Step 1 Complete.")


# ==============================================================================
# PART 2: FEATURE PREP FUNCTIONS
# ==============================================================================


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
    df.drop(columns=["seller_cum_count", "global_cum_count"], inplace=True)
    return df


def s2_create_previous_sale_features(df, n_sales_back):
    print(f"Generating {n_sales_back} lag features...")
    df = df.sort_values(["universal_gemrate_id", "grade", "date"]).reset_index(
        drop=True
    )

    feature_cols = [
        "price",
        "half_grade",
        "grade_co_BGS",
        "grade_co_CGC",
        "grade_co_PSA",
    ]
    new_columns = []

    for n in range(1, n_sales_back + 1):
        suffix = f"prev_{n}"
        for col in feature_cols:
            new_col = f"{suffix}_{col}"
            df[new_col] = df.groupby(["universal_gemrate_id", "grade"])[col].shift(n)
            new_columns.append(new_col)

        days_col = f"{suffix}_days_ago"
        prev_date = df.groupby(["universal_gemrate_id", "grade"])["date"].shift(n)
        df[days_col] = (df["date"] - prev_date).dt.days
        new_columns.append(days_col)

    return df, new_columns


def s2_create_lookback_features(df, weeks_back_list):
    print("Generating lookback features...")
    df = df.sort_values(["universal_gemrate_id", "grade", "date"]).reset_index(
        drop=True
    )
    df["week_start"] = df["date"].dt.to_period("W").dt.start_time

    agg_cols = [
        "price",
        "half_grade",
        "seller_popularity",
        "grade_co_BGS",
        "grade_co_CGC",
        "grade_co_PSA",
    ]
    weekly_agg = (
        df.groupby(["universal_gemrate_id", "grade", "week_start"])[agg_cols]
        .mean()
        .reset_index()
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
            shifted[["universal_gemrate_id", "grade", "join_week"] + final_cols],
            left_on=["universal_gemrate_id", "grade", "week_start"],
            right_on=["universal_gemrate_id", "grade", "join_week"],
            how="left",
        ).drop(columns=["join_week"])

    return result_df, new_columns


def s2_create_adjacent_grade_features(df, n_sales_back):
    print("Generating adjacent grade features...")
    df = df.sort_values(["universal_gemrate_id", "grade", "date"]).reset_index(
        drop=True
    )

    feature_cols = [
        "price",
        "half_grade",
        "grade_co_BGS",
        "grade_co_CGC",
        "grade_co_PSA",
        "seller_popularity",
    ]
    new_columns = []

    ref = df[["universal_gemrate_id", "grade", "date"] + feature_cols].copy()
    ref["_sale_idx"] = ref.groupby(["universal_gemrate_id", "grade"]).cumcount()

    for direction, grade_offset in [("above", 1), ("below", -1)]:
        target_grade = (df["grade"] + grade_offset).clip(lower=1.0, upper=10.0)

        df_lookup = df[["universal_gemrate_id", "date"]].copy()
        df_lookup["target_grade"] = target_grade
        df_lookup["_orig_row"] = df.index
        df_lookup = df_lookup.sort_values("date")

        ref_lookup = ref.rename(
            columns={"grade": "target_grade", "date": "ref_date"}
        ).sort_values("ref_date")

        merged = pd.merge_asof(
            df_lookup,
            ref_lookup[
                ["universal_gemrate_id", "target_grade", "ref_date", "_sale_idx"]
            ],
            left_on="date",
            right_on="ref_date",
            by=["universal_gemrate_id", "target_grade"],
            direction="backward",
            allow_exact_matches=False,
        )

        for n in range(1, n_sales_back + 1):
            suffix = f"prev_{n}_{direction}"
            merged["_lookup_idx"] = merged["_sale_idx"] - (n - 1)

            ref_feats = ref[
                ["universal_gemrate_id", "grade", "_sale_idx", "date"] + feature_cols
            ].copy()
            ref_feats.columns = [
                "universal_gemrate_id",
                "target_grade",
                "_lookup_idx",
                "prev_date",
            ] + feature_cols

            with_feats = merged.merge(
                ref_feats,
                on=["universal_gemrate_id", "target_grade", "_lookup_idx"],
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
    idx_df["index_ema_12"] = idx_df["index_value"].ewm(span=12).mean()
    idx_df["index_ema_26"] = idx_df["index_value"].ewm(span=26).mean()

    df = df.merge(idx_df, on="date", how="left")
    return df


def s2_load_and_clean_ebay(filepath):
    """Load and clean eBay data."""
    print(f"Loading eBay data from {filepath}...")
    df = pd.read_csv(filepath)

    df["date"] = pd.to_datetime(df["item_data.date"], errors="coerce")
    df = df.dropna(subset=["date"])

    df["universal_gemrate_id"] = df["gemrate_data.universal_gemrate_id"].astype(str)
    df = df[df["universal_gemrate_id"].notna() & (df["universal_gemrate_id"] != "")]

    df["grade"] = df["gemrate_data.grade"].apply(s2_clean_grade)
    df = df.dropna(subset=["grade"])
    df[["grade", "half_grade"]] = df["grade"].apply(
        lambda x: pd.Series(s2_process_grade(x))
    )

    # Price cleaning (filter USD only)
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

    df["number_of_bids"] = df["item_data.number_of_bids"].fillna(0).astype(int)
    df = df[df["number_of_bids"] >= S2_NUMBER_OF_BIDS_FILTER]

    df["grading_company"] = df["grading_company"].fillna("Unknown")
    df["seller_name"] = df["item_data.seller_name"].fillna("")
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

    # Handle None values in date
    df["api_response.soldDate"] = df["api_response.soldDate"].replace({None: pd.NaT})
    df = df.dropna(subset=["api_response.soldDate"])
    # Strip timezone abbreviation (PDT, PST, etc.)
    df["api_response.soldDate"] = (
        df["api_response.soldDate"]
        .astype(str)
        .str.replace(r" [A-Z]{3,4}$", "", regex=True)
    )
    df["date"] = pd.to_datetime(df["api_response.soldDate"], errors="coerce")
    df = df.dropna(subset=["date"])

    df["universal_gemrate_id"] = df["gemrate_data.universal_gemrate_id"].astype(str)
    df = df[df["universal_gemrate_id"].notna() & (df["universal_gemrate_id"] != "")]

    df["grade"] = df["gemrate_data.grade"].apply(s2_clean_grade)
    df = df.dropna(subset=["grade"])
    df[["grade", "half_grade"]] = df["grade"].apply(
        lambda x: pd.Series(s2_process_grade(x))
    )

    # Price (PWCC is already USD numeric)
    df["price"] = pd.to_numeric(df["api_response.purchasePrice"], errors="coerce")
    df = df.dropna(subset=["price"])
    df["price"] = np.log(df["price"].clip(lower=0.01))

    df["number_of_bids"] = 0  # PWCC doesn't have bid count
    df["grading_company"] = df["api_response.gradingService"].fillna("Unknown")
    df["seller_name"] = "fanatics"
    df["source"] = "fanatics"

    print(f"  → PWCC cleaned: {len(df)} rows")
    return df


def run_step_2():
    print("Starting Step 2: Feature Prep...")

    # Load both data sources
    ebay_df = s2_load_and_clean_ebay(S1_EBAY_MARKET_FILE)
    pwcc_df = s2_load_and_clean_pwcc(S1_PWCC_MARKET_FILE)

    # Common columns to keep for merge
    common_cols = [
        "universal_gemrate_id",
        "date",
        "price",
        "grade",
        "half_grade",
        "number_of_bids",
        "grading_company",
        "seller_name",
        "source",
    ]

    ebay_subset = ebay_df[[c for c in common_cols if c in ebay_df.columns]].copy()
    pwcc_subset = pwcc_df[[c for c in common_cols if c in pwcc_df.columns]].copy()

    # Merge sources
    print("Merging eBay and PWCC data...")
    df = pd.concat([ebay_subset, pwcc_subset], ignore_index=True)
    print(f"  → Total merged: {len(df)} rows")
    print(f"  → Source distribution: {df['source'].value_counts().to_dict()}")

    # Add source one-hot encoding
    df["source_ebay"] = (df["source"] == "ebay").astype(int)
    df["source_fanatics"] = (df["source"] == "fanatics").astype(int)

    # Create grading company dummies
    dummies = pd.get_dummies(df["grading_company"], prefix="grade_co", dtype=int)
    df = pd.concat([df, dummies], axis=1)

    # Calculate seller popularity (across all sources)
    df = s2_calculate_seller_popularity(df)

    # Columns to keep
    columns_to_keep = [
        "universal_gemrate_id",
        "date",
        "price",
        "grade",
        "half_grade",
        "number_of_bids",
        "source_ebay",
        "source_fanatics",
        "seller_popularity",
    ]
    columns_to_keep.extend(dummies.columns.tolist())
    df = df[columns_to_keep].copy()

    # Feature engineering (grouped by universal_gemrate_id + grade across all sources)
    df, prev_cols = s2_create_previous_sale_features(df, S2_N_SALES_BACK)
    df, lookback_cols = s2_create_lookback_features(df, S2_WEEKS_BACK_LIST)
    df, adj_cols = s2_create_adjacent_grade_features(df, S2_N_SALES_BACK)
    df = s2_load_and_join_index_data(df, S1_INDEX_FILE)

    df = df.drop("week_start", axis=1)
    print(f"Saving {len(df)} rows to {S2_FEATURES_PREPPED_FILE}...")
    df.to_csv(S2_FEATURES_PREPPED_FILE, index=False)
    print("Step 2 Complete.")


# ==============================================================================
# PART 3: TRAIN EMBEDDING FUNCTIONS & CLASSES
# ==============================================================================


# File to load combined cards from (run explore_fill_missing_cards.py first)
S3_ALL_CARDS_FILE = "all_cards_for_embedding.csv"


def s3_load_cards_from_file_or_mongo():
    """Load cards from pre-built CSV (preferred) or fallback to MongoDB."""
    GEMRATE_ID_COL = "GEMRATE_ID"
    METADATA_COLS = [
        GEMRATE_ID_COL,
        "CATEGORY",
        "YEAR",
        "SET_NAME",
        "NAME",
        "PARALLEL",
        "CARD_NUMBER",
    ]

    # Prefer loading from combined file (includes cards from sales data)
    if os.path.exists(S3_ALL_CARDS_FILE):
        print(f"Loading cards from {S3_ALL_CARDS_FILE}...")
        cards = pd.read_csv(S3_ALL_CARDS_FILE)
        cards[GEMRATE_ID_COL] = cards[GEMRATE_ID_COL].astype(str)
        cards["YEAR"] = pd.to_numeric(cards["YEAR"], errors="coerce")
        cards = cards.drop_duplicates(subset=[GEMRATE_ID_COL])
        print(f"  → Loaded {len(cards)} cards from file")
        return cards

    # Fallback to MongoDB
    print(f"Warning: {S3_ALL_CARDS_FILE} not found, loading from MongoDB...")
    print("  → Run explore_fill_missing_cards.py to build complete card list")

    mongo_url = os.getenv("MONGO_URL") or os.getenv("MONGO_URI")
    if not mongo_url:
        print("Skipping Mongo load (MONGO_URL not set).")
        return pd.DataFrame()

    client = MongoClient(mongo_url)
    db = client[S1_DB_NAME]
    print("Loading cards from gemrate_pokemon_cards...")
    cards_collection = db[S3_CARD_COLLECTION]
    cards = pd.json_normalize(list(cards_collection.aggregate([])))

    if cards.empty:
        return cards

    cards = cards[METADATA_COLS].copy()
    cards[GEMRATE_ID_COL] = cards[GEMRATE_ID_COL].astype(str)
    cards["YEAR"] = pd.to_numeric(cards["YEAR"], errors="coerce")
    return cards.drop_duplicates(subset=[GEMRATE_ID_COL])


def s3_to_token_lists(X: np.ndarray) -> list[list[str]]:
    vals = pd.Series(np.ravel(X)).fillna("").astype(str)
    return [[f"card_number={v}"] for v in vals]


def s3_get_preprocess_pipeline():
    CATEGORICAL_COLS = ["CATEGORY", "SET_NAME", "PARALLEL"]
    NUMERICAL_COLS = ["YEAR"]
    EMBEDDING_COLS = ["NAME"]
    HASHING_COLS = ["CARD_NUMBER"]

    sparse_block = ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=S3_OHE_MIN_FREQUENCY,
                    sparse_output=True,
                    dtype=np.float32,
                ),
                CATEGORICAL_COLS,
            ),
            (
                "card",
                Pipeline(
                    [
                        (
                            "to_tokens",
                            FunctionTransformer(s3_to_token_lists, validate=False),
                        ),
                        (
                            "hasher",
                            FeatureHasher(
                                n_features=S3_HASH_N_FEATURES,
                                input_type="string",
                                alternate_sign=False,
                            ),
                        ),
                    ]
                ),
                HASHING_COLS,
            ),
        ],
        remainder="drop",
    )

    sparse_to_dense = Pipeline(
        [
            ("sparse", sparse_block),
            (
                "svd",
                TruncatedSVD(
                    n_components=S3_SVD_N_COMPONENTS, random_state=S3_RANDOM_SEED
                ),
            ),
            (
                "to32",
                FunctionTransformer(lambda X: X.astype(np.float32), validate=False),
            ),
        ]
    )

    embed_model = SentenceTransformer("BAAI/bge-m3")

    def embed_text_batch(X):
        texts = pd.Series(np.ravel(X)).fillna("").astype(str).tolist()
        emb = embed_model.encode(texts, batch_size=64, show_progress_bar=True)
        return np.asarray(emb, dtype=np.float32)

    dense_block = ColumnTransformer(
        [
            (
                "year",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                        (
                            "to32",
                            FunctionTransformer(
                                lambda X: X.astype(np.float32), validate=False
                            ),
                        ),
                    ]
                ),
                NUMERICAL_COLS,
            ),
            (
                "name",
                FunctionTransformer(embed_text_batch, validate=False),
                EMBEDDING_COLS,
            ),
        ],
        remainder="drop",
    )

    return Pipeline(
        [
            (
                "union",
                FeatureUnion(
                    [("meta_lowdim", sparse_to_dense), ("dense", dense_block)]
                ),
            )
        ]
    )


def s3_build_weekly_sales_vectors_global_time(sales_df):
    """Build grade-agnostic sales vectors - aggregate all grades per card."""
    print("Building Global-Time Sales Vectors (Grade-Agnostic)...")
    s = sales_df.copy()
    s["date"] = pd.to_datetime(s["date"], format="mixed", utc=True)

    global_start = pd.Timestamp("2018-01-01", tz="UTC")

    s["week"] = (s["date"] - global_start).dt.days // 7
    s = s.loc[s["week"] >= 0]
    n_weeks = s["week"].max()

    # Group by card only (ignore grade) - aggregate all sales for a card
    group_cols = ["universal_gemrate_id"]
    s["log_price"] = np.log1p(s["price"])

    price_wide = s.pivot_table(
        index=group_cols, columns="week", values="log_price", aggfunc="median"
    )
    price_wide = price_wide.reindex(columns=range(n_weeks)).astype(np.float32)

    vol_wide = s.pivot_table(
        index=group_cols, columns="week", values="log_price", aggfunc="size"
    )
    vol_wide = vol_wide.reindex(columns=range(n_weeks)).fillna(0.0).astype(np.float32)

    P = price_wide.to_numpy()
    M = (~np.isnan(P)).astype(np.float32)
    P0 = np.nan_to_num(P, nan=0.0)

    denom = np.maximum(M.sum(axis=1, keepdims=True), 1.0)
    mu = (P0 * M).sum(axis=1, keepdims=True) / denom
    var = (((P0 - mu) * M) ** 2).sum(axis=1, keepdims=True) / denom
    sigma = np.sqrt(var).astype(np.float32)
    sigma = np.maximum(sigma, 1e-3)
    Pz = ((P0 - mu) / sigma) * M

    V = np.minimum(vol_wide.to_numpy(), S3_VOLUME_CAP_PER_WEEK)
    V_log = np.log1p(V)
    total_volume = V.sum(axis=1, keepdims=True)

    S = np.concatenate(
        [Pz, M, V_log, mu, sigma, np.log1p(total_volume)], axis=1
    ).astype(np.float32)

    # Keys are now just card IDs (no grade)
    keys = pd.DataFrame(price_wide.index.to_list(), columns=["universal_gemrate_id"])

    return keys, S, M.sum(axis=1)


class Encoder(nn.Module):
    """Grade-agnostic encoder - takes only card metadata, no grade input."""

    def __init__(self, meta_dim, out_dim=S3_ENCODER_OUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(meta_dim, S3_ENCODER_HIDDEN_1),
            nn.ReLU(),
            nn.Dropout(S3_DROPOUT_P),
            nn.Linear(S3_ENCODER_HIDDEN_1, S3_ENCODER_HIDDEN_2),
            nn.ReLU(),
            nn.Dropout(S3_DROPOUT_P),
            nn.Linear(S3_ENCODER_HIDDEN_2, out_dim),
        )

    def forward(self, x_meta):
        z = self.net(x_meta)
        return F.normalize(z, p=2, dim=1)


def s3_cosine_sim_matrix(x):
    x = F.normalize(x, p=2, dim=1)
    return x @ x.T


def s3_compute_batch_loss(enc, xb_meta, sb, wb, tau):
    """Compute KL-divergence loss between embedding similarity and sales similarity."""
    w = torch.clamp(wb, min=0.0)
    w = w / (w.sum() + 1e-8)

    z = enc(xb_meta)
    Kz = s3_cosine_sim_matrix(z)
    Ks = s3_cosine_sim_matrix(sb)

    b = z.size(0)
    diag = torch.eye(b, device=z.device).bool()
    Kz = Kz.masked_fill(diag, -1e9)
    Ks = Ks.masked_fill(diag, -1e9)

    logP = F.log_softmax(Kz / tau, dim=1)
    Q = F.softmax(Ks / tau, dim=1)

    row_kl = F.kl_div(logP, Q, reduction="none").sum(dim=1)
    loss = (row_kl * w).sum() * (tau * tau)
    return loss


def run_step_3():
    """Train grade-agnostic card embeddings using sales history similarity."""
    print("Starting Step 3: Train Embedding (Grade-Agnostic)...")
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    cards = s3_load_cards_from_file_or_mongo()
    if cards.empty:
        raise ValueError("No cards found.")

    print("Processing metadata...")
    preprocess = s3_get_preprocess_pipeline()
    X_all = preprocess.fit_transform(cards).astype(np.float32)
    X_all = np.nan_to_num(X_all)

    if not os.path.exists(S2_FEATURES_PREPPED_FILE):
        raise FileNotFoundError(f"{S2_FEATURES_PREPPED_FILE} not found.")

    sales_df = pd.read_csv(S2_FEATURES_PREPPED_FILE)
    sales_df["universal_gemrate_id"] = sales_df["universal_gemrate_id"].astype(str)

    # Build grade-agnostic sales vectors (one per card, aggregating all grades)
    sales_keys, S, obs_weeks = s3_build_weekly_sales_vectors_global_time(sales_df)

    # Align sales keys to card metadata
    id_to_row = {str(gid): i for i, gid in enumerate(cards["GEMRATE_ID"])}
    meta_feats, S_aligned, obs_aligned, aligned_keys = [], [], [], []

    for i in range(len(sales_keys)):
        gid = str(sales_keys.iloc[i]["universal_gemrate_id"])
        if gid in id_to_row:
            idx = id_to_row[gid]
            meta_feats.append(X_all[idx])
            S_aligned.append(S[i])
            obs_aligned.append(obs_weeks[i])
            aligned_keys.append({"universal_gemrate_id": gid})

    print(f"Aligned {len(meta_feats)} cards with sales to metadata.")
    X_meta = np.stack(meta_feats)
    S_final = np.stack(S_aligned)
    W_final = np.array(obs_aligned).astype(np.float32)
    keys_df = pd.DataFrame(aligned_keys)

    print("Performing train/val split...")
    gss = GroupShuffleSplit(
        n_splits=1, test_size=S3_FINETUNE_VAL_SPLIT, random_state=S3_RANDOM_SEED
    )
    train_idx, val_idx = next(gss.split(X_meta, groups=keys_df["universal_gemrate_id"]))
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)}")

    enc = Encoder(meta_dim=X_meta.shape[1]).to(device)
    opt = torch.optim.AdamW(enc.parameters(), lr=S3_FINETUNE_LR)

    # Datasets without grade (only meta, sales, weights)
    train_dataset = TensorDataset(
        torch.from_numpy(X_meta[train_idx]),
        torch.from_numpy(S_final[train_idx]),
        torch.from_numpy(W_final[train_idx]),
    )

    val_dataset = TensorDataset(
        torch.from_numpy(X_meta[val_idx]),
        torch.from_numpy(S_final[val_idx]),
        torch.from_numpy(W_final[val_idx]),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=S3_FINETUNE_BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=S3_FINETUNE_BATCH_SIZE, shuffle=False, drop_last=False
    )

    # Early stopping and checkpointing
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_encoder_state = None

    print(f"Starting Finetuning on {device}...")
    print(f"  → Early stopping patience: {S3_FINETUNE_PATIENCE} epochs")

    for epoch in range(S3_FINETUNE_EPOCHS):
        # Training
        enc.train()
        train_loss = 0.0

        for batch_meta, batch_s, batch_w in train_loader:
            b_meta = batch_meta.to(device)
            b_s = batch_s.to(device)
            b_w = batch_w.to(device)

            loss = s3_compute_batch_loss(enc, b_meta, b_s, b_w, S3_FINETUNE_TAU)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        enc.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_meta, batch_s, batch_w in val_loader:
                b_meta = batch_meta.to(device)
                b_s = batch_s.to(device)
                b_w = batch_w.to(device)

                loss = s3_compute_batch_loss(enc, b_meta, b_s, b_w, S3_FINETUNE_TAU)
                val_loss += loss.item()

        avg_val_loss = (
            val_loss / len(val_loader) if len(val_loader) > 0 else float("inf")
        )

        # Check for improvement
        improved = avg_val_loss < best_val_loss
        if improved:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            best_encoder_state = {
                k: v.cpu().clone() for k, v in enc.state_dict().items()
            }
            marker = " ✓ (best)"
        else:
            epochs_without_improvement += 1
            marker = f" (no improvement: {epochs_without_improvement}/{S3_FINETUNE_PATIENCE})"

        print(
            f"Epoch {epoch + 1}: Train {avg_train_loss:.4f} | Val {avg_val_loss:.4f}{marker}"
        )

        # Early stopping check
        if epochs_without_improvement >= S3_FINETUNE_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # Load best model
    if best_encoder_state is not None:
        enc.load_state_dict(best_encoder_state)
        print(f"Loaded best model (val loss: {best_val_loss:.4f})")

    torch.save(enc.state_dict(), "final_encoder.pt")
    print("Saved final_encoder.pt")

    # Generate one embedding per card (grade-agnostic)
    print("Generating card embeddings (one per card, grade-agnostic)...")
    enc.eval()
    final_embs = []
    final_keys = []

    with torch.no_grad():
        for i in range(0, len(X_all), S3_EMBED_BATCH_SIZE):
            batch = torch.from_numpy(X_all[i : i + S3_EMBED_BATCH_SIZE]).to(device)
            emb = enc(batch).cpu().numpy()
            final_embs.append(emb)

    all_embs = np.vstack(final_embs)

    for _, r in cards.iterrows():
        final_keys.append({"universal_gemrate_id": str(r["GEMRATE_ID"])})

    np.save(S3_EMBEDDING_VECTORS_FILE, all_embs)
    pd.DataFrame(final_keys).to_csv(S3_EMBEDDING_KEYS_FILE, index=False)
    print(f"Saved {len(all_embs)} embeddings (one per card).")
    print("Step 3 Complete.")


# ==============================================================================
# PART 4: MERGE PREDICTIONS FUNCTIONS
# ==============================================================================


def run_step_4():
    """
    Find similar cards using grade-agnostic embeddings, then filter by grade at query time.

    For each sale row:
    1. Look up the card's embedding (grade-agnostic)
    2. Find top-K most similar cards by embedding
    3. For each neighbor, look up sales at the SAME GRADE as the target row
    4. Return neighbor prices from same-grade sales in lookback window
    """
    print("Starting Step 4: Merge Embedding Predictions (Grade-Filtered)...")
    print("Loading data...")
    if not os.path.exists(S2_FEATURES_PREPPED_FILE):
        raise FileNotFoundError(
            f"{S2_FEATURES_PREPPED_FILE} not found. Run step 2 first."
        )

    df = pd.read_csv(S2_FEATURES_PREPPED_FILE)
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    df["universal_gemrate_id"] = df["universal_gemrate_id"].astype(str)

    if not os.path.exists(S3_EMBEDDING_KEYS_FILE) or not os.path.exists(
        S3_EMBEDDING_VECTORS_FILE
    ):
        raise FileNotFoundError("Embedding files not found. Run step 3 first.")

    keys = pd.read_csv(S3_EMBEDDING_KEYS_FILE)
    keys["universal_gemrate_id"] = keys["universal_gemrate_id"].astype(str)

    all_vecs = np.load(S3_EMBEDDING_VECTORS_FILE).astype(np.float32)

    print(f"Loaded {len(df)} sales rows and {len(all_vecs)} card embeddings.")

    # Build card_id -> vec_idx mapping (grade-agnostic)
    card_to_vec_idx = {
        str(keys.iloc[i]["universal_gemrate_id"]): i for i in range(len(keys))
    }

    # Build sales index by (card_id, grade)
    print("Indexing sales history by (card, grade)...")
    df_sorted = df.sort_values("date")
    sales_lookup = {}
    grouped = df_sorted.groupby(["universal_gemrate_id", "grade"])

    for (sid, g), group in tqdm(grouped, desc="Building sales index"):
        dates = group["date"].values
        prices = group["price"].values
        sales_lookup[(str(sid), float(g))] = (dates, prices)

    # Get unique cards that have sales
    cards_with_sales = list(set(k[0] for k in sales_lookup.keys()))
    print(f"Found {len(cards_with_sales)} unique cards with sales.")

    # Extract vectors only for cards with sales
    sales_card_indices = []
    valid_cards = []
    for cid in cards_with_sales:
        if cid in card_to_vec_idx:
            sales_card_indices.append(card_to_vec_idx[cid])
            valid_cards.append(cid)

    print(f"Matched {len(valid_cards)} cards to embeddings.")
    sales_vecs = all_vecs[sales_card_indices]

    # Normalize
    print("Normalizing vectors...")
    norms = np.linalg.norm(sales_vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    sales_vecs = sales_vecs / norms

    # Build card_id -> index in sales_vecs
    card_to_sales_idx = {cid: i for i, cid in enumerate(valid_cards)}

    # Pre-compute top-K neighbor cards for each card with sales
    print(
        f"Pre-computing top {S4_K_CANDIDATES} neighbor cards for {len(sales_vecs)} cards..."
    )

    n_cards = len(sales_vecs)
    neighbor_card_indices = np.zeros((n_cards, S4_K_CANDIDATES), dtype=np.int32)
    neighbor_card_sims = np.zeros((n_cards, S4_K_CANDIDATES), dtype=np.float32)

    batch_size = 1000
    for i in tqdm(range(0, n_cards, batch_size), desc="Computing neighbors"):
        end = min(i + batch_size, n_cards)
        batch = sales_vecs[i:end]

        # Compute similarities to ALL cards with sales
        sims = batch @ sales_vecs.T

        # Zero out self-similarity
        for j in range(end - i):
            sims[j, i + j] = -1.0

        # Get top K
        top_k_idx = np.argpartition(-sims, S4_K_CANDIDATES, axis=1)[:, :S4_K_CANDIDATES]

        # Sort within top K
        for j in range(end - i):
            sorted_order = np.argsort(-sims[j, top_k_idx[j]])
            neighbor_card_indices[i + j] = top_k_idx[j][sorted_order]
            neighbor_card_sims[i + j] = sims[j, top_k_idx[j][sorted_order]]

    print(f"Finding {S4_N_NEIGHBORS} same-grade neighbors for each sale...")

    n_rows = len(df)
    res_prices = np.full((n_rows, S4_N_NEIGHBORS), np.nan, dtype=np.float32)
    res_sims = np.full((n_rows, S4_N_NEIGHBORS), np.nan, dtype=np.float32)

    row_sids = df["universal_gemrate_id"].values
    row_grades = df["grade"].values
    row_dates = df["date"].values

    lookback_delta = np.timedelta64(S4_LOOKBACK_DAYS, "D")

    for i in tqdm(range(n_rows), desc="Processing rows"):
        s_id = str(row_sids[i])
        target_grade = float(row_grades[i])
        dt = row_dates[i]

        if s_id not in card_to_sales_idx:
            continue

        card_idx = card_to_sales_idx[s_id]

        # Get neighbor cards (by embedding similarity)
        candidate_card_indices = neighbor_card_indices[card_idx]
        candidate_card_sims = neighbor_card_sims[card_idx]

        found_count = 0
        window_start = dt - lookback_delta

        # For each neighbor card, check if it has sales at the TARGET GRADE
        for rank, neighbor_card_idx in enumerate(candidate_card_indices):
            neighbor_card_id = valid_cards[neighbor_card_idx]

            # Look up sales for this neighbor card at the TARGET GRADE
            neighbor_key = (neighbor_card_id, target_grade)

            if neighbor_key not in sales_lookup:
                continue  # This neighbor card has no sales at this grade

            n_dates, n_prices = sales_lookup[neighbor_key]

            # Find sales in lookback window
            start_idx = np.searchsorted(n_dates, window_start, side="left")
            end_idx = np.searchsorted(n_dates, dt, side="right")

            if end_idx > start_idx:
                avg_price = np.mean(n_prices[start_idx:end_idx])

                res_prices[i, found_count] = avg_price
                res_sims[i, found_count] = candidate_card_sims[rank]

                found_count += 1

                if found_count >= S4_N_NEIGHBORS:
                    break

    print("Appending new columns...")
    for n in range(S4_N_NEIGHBORS):
        df[f"neighbor_{n+1}_avg_price"] = res_prices[:, n]
        df[f"neighbor_{n+1}_similarity"] = res_sims[:, n]

    print(f"Saving to {S4_OUTPUT_FILE}...")
    df.to_csv(S4_OUTPUT_FILE, index=False)

    filled = (~np.isnan(res_prices[:, 0])).sum()
    print(
        f"Processing complete. {filled}/{n_rows} rows found at least one same-grade neighbor."
    )
    print("Step 4 Complete.")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    if RUN_STEP_1_DOWNLOAD:
        run_step_1()

    if RUN_STEP_2_FEATURE_PREP:
        run_step_2()

    if RUN_STEP_3_TRAIN_EMBEDDING:
        run_step_3()

    if RUN_STEP_4_MERGE_PREDICTIONS:
        run_step_4()
