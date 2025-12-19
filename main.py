import os
import re
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
import faiss
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
RUN_STEP_1_DOWNLOAD = True
RUN_STEP_2_FEATURE_PREP = True
RUN_STEP_3_TRAIN_EMBEDDING = True
RUN_STEP_4_MERGE_PREDICTIONS = True

# Part 1: Download Sales Config
S1_MONGO_URL = os.getenv("MONGO_URL")
S1_DB_NAME = "gemrate"
S1_MARKET_INFO_COLLECTION = "ebay_graded_items"
S1_INDEX_API_URL = "https://price.collectorcrypt.com/api/indexes/modern"
S1_MARKET_FILE = "market.csv"
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
    collection = db[S1_MARKET_INFO_COLLECTION]

    pipeline = [
        {
            "$match": {
                "gemrate_data.universal_gemrate_id": {"$exists": True},
                "gemrate_hybrid_data.specid": {"$exists": True},
                "item_data.format": "auction",
                "gemrate_hybrid_data": {"$exists": True},
                "item_data": {"$exists": True},
                "gemrate_data": {"$exists": True},
            }
        },
        {
            "$project": {
                "gemrate_data.universal_gemrate_id": 1,
                "gemrate_hybrid_data.specid": 1,
                "item_data.date": 1,
                "grading_company": 1,
                "gemrate_data.grade": 1,
                "item_data.price": 1,
                "item_data.number_of_bids": 1,
                "item_data.seller_name": 1,
                "item_data.best_offer_accepted": 1,
                "_id": 1,
            }
        },
    ]

    results = collection.aggregate(pipeline, maxTimeMS=6000000, allowDiskUse=True)
    df = pd.DataFrame(list(results))
    df = pd.json_normalize(df.to_dict("records"))
    print(f"Processing complete. Rows loaded: {len(df)}")
    df.to_csv(S1_MARKET_FILE, index=False)

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
    if s.startswith("$") or s[0].isdigit():
        return "$ (No Country Code)"
    return s

def s2_calculate_seller_popularity(df):
    print("Calculating seller popularity (expanding window)...")
    df = df.sort_values("date").copy()
    df["seller_cum_count"] = df.groupby("item_data.seller_name").cumcount() + 1
    df["global_cum_count"] = np.arange(1, len(df) + 1)
    df["seller_popularity"] = df["seller_cum_count"] / df["global_cum_count"]
    df.drop(columns=["seller_cum_count", "global_cum_count"], inplace=True)
    return df

def s2_create_previous_sale_features(df, n_sales_back):
    print(f"Generating {n_sales_back} lag features...")
    df = df.sort_values(["universal_gemrate_id", "grade", "date"]).reset_index(drop=True)

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
    df = df.sort_values(["universal_gemrate_id", "grade", "date"]).reset_index(drop=True)
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
    df = df.sort_values(["universal_gemrate_id", "grade", "date"]).reset_index(drop=True)

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

def run_step_2():
    print("Starting Step 2: Feature Prep...")
    if not os.path.exists(S1_MARKET_FILE):
        raise FileNotFoundError(f"{S1_MARKET_FILE} not found. Run Step 1 first.")

    print("Loading market data...")
    df = pd.read_csv(S1_MARKET_FILE)

    df["date"] = pd.to_datetime(df["item_data.date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)

    df["universal_gemrate_id"] = df["gemrate_data.universal_gemrate_id"]
    df["spec_id"] = df["gemrate_hybrid_data.specid"]
    df = df.dropna(subset=["universal_gemrate_id", "spec_id"])

    df["grade"] = df["gemrate_data.grade"].apply(s2_clean_grade)
    df = df.dropna(subset=["grade"])
    df[["grade", "half_grade"]] = df["grade"].apply(
        lambda x: pd.Series(s2_process_grade(x))
    )

    currency_groups = df["item_data.price"].str.split().str[0].apply(s2_group_currencies)
    df = df.loc[currency_groups.isin(["$ (No Country Code)", "US"])]
    df["price"] = np.log(
        df["item_data.price"]
        .astype(str)
        .str.replace(r"\D+", "", regex=True)
        .astype(float)
    )

    df["number_of_bids"] = df["item_data.number_of_bids"].fillna(0).astype(int)
    df = df[df["number_of_bids"] >= S2_NUMBER_OF_BIDS_FILTER]

    df["grading_company"] = df["grading_company"].fillna("Unknown")
    dummies = pd.get_dummies(df["grading_company"], prefix="grade_co", dtype=int)
    df = pd.concat([df, dummies], axis=1)

    columns_to_keep = [
        "universal_gemrate_id",
        "spec_id",
        "date",
        "price",
        "grade",
        "half_grade",
        "number_of_bids",
    ]
    columns_to_keep.extend(dummies.columns.tolist())

    df = s2_calculate_seller_popularity(df)
    columns_to_keep.append("seller_popularity")

    df = df[columns_to_keep].copy()

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

def s3_load_cards_from_mongo():
    print("Connecting to MongoDB...")
    if "MONGO_URL" not in os.environ:
        print("Skipping Mongo load (MONGO_URL not set).")
        return pd.DataFrame()

    client = MongoClient(os.environ["MONGO_URL"])
    db = client[S1_DB_NAME]
    print("Loading cards...")
    cards_collection = db[S3_CARD_COLLECTION]
    cards = pd.json_normalize(list(cards_collection.aggregate([])))

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
                TruncatedSVD(n_components=S3_SVD_N_COMPONENTS, random_state=S3_RANDOM_SEED),
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
    print("Building Global-Time Sales Vectors...")
    s = sales_df.copy()
    s["date"] = pd.to_datetime(s["date"], utc=True)

    global_start = pd.Timestamp("2018-01-01", tz="UTC")

    s["week"] = (s["date"] - global_start).dt.days // 7
    s = s.loc[s["week"] >= 0]
    n_weeks = s["week"].max()

    group_cols = ["universal_gemrate_id", "grade"]
    s["log_price"] = np.log1p(s["price"])

    price_wide = s.pivot_table(
        index=group_cols, columns="week", values="log_price", aggfunc="median"
    )
    price_wide = price_wide.reindex(columns=range(n_weeks)).astype(np.float32)

    vol_wide = s.pivot_table(
        index=group_cols, columns="week", values="log_price", aggfunc="size"
    )
    vol_wide = vol_wide.reindex(columns=range(n_weeks)).fillna(0.0).astype(np.float32)

    half_grade_map = s.groupby(group_cols)["half_grade"].first()

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

    keys = pd.DataFrame(price_wide.index.to_list(), columns=["universal_gemrate_id", "grade"])
    keys["half_grade"] = half_grade_map.loc[
        keys.set_index(["universal_gemrate_id", "grade"]).index
    ].values

    return keys, S, M.sum(axis=1)

def s3_normalize_grade(grade, half_grade):
    g = (grade - 1.0) / 10.0
    return np.array([g, half_grade], dtype=np.float32)

class GradeProjector(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, out_dim=S3_GRADE_EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        return self.net(x)

class Encoder(nn.Module):
    def __init__(self, meta_dim, grade_dim=2, out_dim=S3_ENCODER_OUT_DIM):
        super().__init__()
        self.grade_proj = GradeProjector(input_dim=grade_dim)
        self.fc_in_dim = meta_dim + S3_GRADE_EMBED_DIM

        self.net = nn.Sequential(
            nn.Linear(self.fc_in_dim, S3_ENCODER_HIDDEN_1),
            nn.ReLU(),
            nn.Dropout(S3_DROPOUT_P),
            nn.Linear(S3_ENCODER_HIDDEN_1, S3_ENCODER_HIDDEN_2),
            nn.ReLU(),
            nn.Dropout(S3_DROPOUT_P),
            nn.Linear(S3_ENCODER_HIDDEN_2, out_dim),
        )

    def forward(self, x_meta, x_grade):
        g_emb = self.grade_proj(x_grade)
        x = torch.cat([x_meta, g_emb], dim=1)
        z = self.net(x)
        return F.normalize(z, p=2, dim=1)

def s3_cosine_sim_matrix(x):
    x = F.normalize(x, p=2, dim=1)
    return x @ x.T

def s3_compute_batch_loss(enc, xb_meta, xb_grade, sb, wb, tau):
    w = torch.clamp(wb, min=0.0)
    w = w / (w.sum() + 1e-8)

    z = enc(xb_meta, xb_grade)
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
    print("Starting Step 3: Train Embedding...")
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    cards = s3_load_cards_from_mongo()
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
    sales_keys, S, obs_weeks = s3_build_weekly_sales_vectors_global_time(sales_df)

    id_to_row = {str(gid): i for i, gid in enumerate(cards["GEMRATE_ID"])}
    meta_feats, grade_feats, S_aligned, obs_aligned, aligned_keys = [], [], [], [], []

    for i in range(len(sales_keys)):
        gid = str(sales_keys.iloc[i]["universal_gemrate_id"])
        if gid in id_to_row:
            idx = id_to_row[gid]
            g = float(sales_keys.iloc[i]["grade"])
            h = float(sales_keys.iloc[i]["half_grade"])
            meta_feats.append(X_all[idx])
            grade_feats.append(s3_normalize_grade(g, h))
            S_aligned.append(S[i])
            obs_aligned.append(obs_weeks[i])
            aligned_keys.append(sales_keys.iloc[i])

    print(f"Aligned {len(meta_feats)} sales records to card metadata.")
    X_meta = np.stack(meta_feats)
    X_grade = np.stack(grade_feats)
    S_final = np.stack(S_aligned)
    W_final = np.array(obs_aligned).astype(np.float32)
    keys_df = pd.DataFrame(aligned_keys)

    print("Performing Stratified Split by universal_gemrate_id...")
    gss = GroupShuffleSplit(
        n_splits=1, test_size=S3_FINETUNE_VAL_SPLIT, random_state=S3_RANDOM_SEED
    )
    train_idx, val_idx = next(gss.split(X_meta, groups=keys_df["universal_gemrate_id"]))
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)}")

    enc = Encoder(meta_dim=X_meta.shape[1]).to(device)
    opt = torch.optim.AdamW(enc.parameters(), lr=S3_FINETUNE_LR)

    train_dataset = TensorDataset(
        torch.from_numpy(X_meta[train_idx]),
        torch.from_numpy(X_grade[train_idx]),
        torch.from_numpy(S_final[train_idx]),
        torch.from_numpy(W_final[train_idx]),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=S3_FINETUNE_BATCH_SIZE, shuffle=True, drop_last=True
    )

    print(f"Starting Finetuning on {device}...")
    for epoch in range(S3_FINETUNE_EPOCHS):
        enc.train()
        total_loss = 0.0

        for batch_meta, batch_grade, batch_s, batch_w in train_loader:
            b_meta = batch_meta.to(device)
            b_grade = batch_grade.to(device)
            b_s = batch_s.to(device)
            b_w = batch_w.to(device)

            loss = s3_compute_batch_loss(enc, b_meta, b_grade, b_s, b_w, S3_FINETUNE_TAU)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}: Avg Loss {avg_loss:.4f}")

    torch.save(enc.state_dict(), "final_encoder.pt")

    print("Generating static card embeddings...")
    enc.eval()
    final_embs = []
    final_keys = []
    standard_grades = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]

    with torch.no_grad():
        for g in standard_grades:
            n = len(X_all)
            g_vec = s3_normalize_grade(float(g), 1.0 if g % 1 else 0.0)
            X_g = np.tile(g_vec, (n, 1))

            for i in range(0, n, S3_EMBED_BATCH_SIZE):
                xm = torch.from_numpy(X_all[i : i + S3_EMBED_BATCH_SIZE]).to(device)
                xg = torch.from_numpy(X_g[i : i + S3_EMBED_BATCH_SIZE]).to(device)

                emb = enc(xm, xg).cpu().numpy()
                final_embs.append(emb)

            for _, r in cards.iterrows():
                final_keys.append(
                    {
                        "universal_gemrate_id": str(r["GEMRATE_ID"]),
                        "grade": g,
                        "half_grade": 1.0 if g % 1 else 0.0,
                    }
                )

    np.save(S3_EMBEDDING_VECTORS_FILE, np.vstack(final_embs))
    pd.DataFrame(final_keys).to_csv(S3_EMBEDDING_KEYS_FILE, index=False)
    print("Saved embeddings.")
    print("Step 3 Complete.")

# ==============================================================================
# PART 4: MERGE PREDICTIONS FUNCTIONS
# ==============================================================================

def run_step_4():
    print("Starting Step 4: Merge Embedding Predictions...")
    print("Loading data...")
    if not os.path.exists(S2_FEATURES_PREPPED_FILE):
        raise FileNotFoundError(f"{S2_FEATURES_PREPPED_FILE} not found. Run step 2 first.")

    df = pd.read_csv(S2_FEATURES_PREPPED_FILE)
    df['date'] = pd.to_datetime(df['date'])
    df['universal_gemrate_id'] = df['universal_gemrate_id'].astype(str)

    if not os.path.exists(S3_EMBEDDING_KEYS_FILE) or not os.path.exists(S3_EMBEDDING_VECTORS_FILE):
        raise FileNotFoundError("Embedding files not found. Run step 3 first.")

    keys = pd.read_csv(S3_EMBEDDING_KEYS_FILE)
    keys['universal_gemrate_id'] = keys['universal_gemrate_id'].astype(str)

    vecs = np.load(S3_EMBEDDING_VECTORS_FILE).astype(np.float32)

    print(f"Loaded {len(df)} sales rows and {len(vecs)} embeddings.")

    keys['key_tuple'] = list(zip(keys['universal_gemrate_id'], keys['grade']))
    key_to_vec_idx = {k: i for i, k in enumerate(keys['key_tuple'])}
    keys_lookup_list = keys['key_tuple'].tolist()

    print("Indexing sales history...")
    df_sorted = df.sort_values('date')
    sales_lookup = {}
    grouped = df_sorted.groupby(['universal_gemrate_id', 'grade'])

    for (sid, g), group in tqdm(grouped, desc="Building sales index"):
        dates = group['date'].values
        prices = group['price'].values
        sales_lookup[(str(sid), float(g))] = (dates, prices)

    print(f"Pre-computing top {S4_K_CANDIDATES} neighbors for {len(vecs)} vectors on GPU...")

    d = vecs.shape[1]

    faiss.normalize_L2(vecs)

    res = faiss.StandardGpuResources()

    index_flat = faiss.IndexFlatIP(d)

    gpu_index = faiss.index_cpu_to_gpu(res, 0, index_flat)

    gpu_index.add(vecs)

    similarities, indices = gpu_index.search(vecs, S4_K_CANDIDATES + 1)

    distances = 1.0 - similarities

    indices = np.array(indices)
    distances = np.array(distances)

    print(f"Finding {S4_N_NEIGHBORS} active neighbors for each sale...")

    n_rows = len(df)
    res_prices = np.full((n_rows, S4_N_NEIGHBORS), np.nan, dtype=np.float32)
    res_sims = np.full((n_rows, S4_N_NEIGHBORS), np.nan, dtype=np.float32)

    row_sids = df['universal_gemrate_id'].values
    row_grades = df['grade'].values
    row_dates = df['date'].values

    lookback_delta = np.timedelta64(S4_LOOKBACK_DAYS, 'D')

    for i in tqdm(range(n_rows), desc="Processing rows"):
        s_id = row_sids[i]
        g = row_grades[i]
        dt = row_dates[i]

        k = (s_id, g)

        if k not in key_to_vec_idx:
            continue

        idx = key_to_vec_idx[k]

        candidate_indices = indices[idx]
        candidate_dists = distances[idx]

        found_count = 0
        window_start = dt - lookback_delta

        for rank, neighbor_idx in enumerate(candidate_indices):
            if neighbor_idx == idx:
                continue

            n_key = keys_lookup_list[neighbor_idx]

            if n_key in sales_lookup:
                n_dates, n_prices = sales_lookup[n_key]

                start_idx = np.searchsorted(n_dates, window_start, side='left')
                end_idx = np.searchsorted(n_dates, dt, side='right')

                if end_idx > start_idx:
                    avg_price = np.mean(n_prices[start_idx:end_idx])

                    sim = 1.0 - candidate_dists[rank]

                    res_prices[i, found_count] = avg_price
                    res_sims[i, found_count] = sim

                    found_count += 1

                    if found_count >= S4_N_NEIGHBORS:
                        break

    print("Appending new columns...")
    for n in range(S4_N_NEIGHBORS):
        df[f'neighbor_{n+1}_avg_price'] = res_prices[:, n]
        df[f'neighbor_{n+1}_similarity'] = res_sims[:, n]

    print(f"Saving to {S4_OUTPUT_FILE}...")
    df.to_csv(S4_OUTPUT_FILE, index=False)

    filled = (~np.isnan(res_prices[:, 0])).sum()
    print(f"Processing complete. {filled}/{n_rows} rows found at least one neighbor.")
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