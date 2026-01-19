"""
Standalone Embedding Model Trainer

This module trains card embeddings using contrastive learning where cards with
similar price behavior over time are mapped to similar embeddings.

Architecture:
- Input: Card metadata (set, name, year, card_number)
- Target: Weekly sales patterns (prices, volumes, observation masks)
- Loss: KL divergence between encoder similarity and sales similarity matrices

Each gemrate_id gets exactly ONE embedding (no grade splitting).

Usage:
    python embedding_trainer.py

Requirements:
    - MongoDB connection (MONGO_URL or MONGO_URI env var)
    - features_prepped.csv (from feature prep step)
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from torch.utils.data import DataLoader, TensorDataset

load_dotenv()

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# MongoDB Config
MONGO_URL = os.getenv("MONGO_URL") or os.getenv("MONGO_URI")
DB_NAME = "gemrate"
CARD_COLLECTION = "pokemon_card_metadata_oracle"

# Input/Output Files
FEATURES_PREPPED_FILE = "features_prepped.csv"
EMBEDDING_KEYS_FILE = "embedding_keys.csv"
EMBEDDING_VECTORS_FILE = "card_vectors_768.npy"
MODEL_CHECKPOINT_FILE = "final_encoder.pt"

# Preprocessing Config
RANDOM_SEED = 42
HASH_N_FEATURES = 2**12  # 4096
SVD_N_COMPONENTS = 256
OHE_MIN_FREQUENCY = 10
VOLUME_CAP_PER_WEEK = 200.0

# Encoder Architecture Config
ENCODER_HIDDEN_1 = 2048
ENCODER_HIDDEN_2 = 1024
ENCODER_OUT_DIM = 768
DROPOUT_P = 0.1

# Training Config
FINETUNE_EPOCHS = 20
FINETUNE_BATCH_SIZE = 512
FINETUNE_LR = 5e-4
FINETUNE_TAU = 0.07  # Temperature for contrastive loss
FINETUNE_VAL_SPLIT = 0.1

# Inference Config
EMBED_BATCH_SIZE = 4096


# ==============================================================================
# DATA LOADING
# ==============================================================================


def load_cards_from_mongo():
    """Load card metadata from MongoDB."""
    print("Connecting to MongoDB...")
    if not MONGO_URL:
        raise ValueError("MONGO_URL or MONGO_URI environment variable not set.")

    client = MongoClient(MONGO_URL)
    db = client[DB_NAME]
    print("Loading cards...")
    cards_collection = db[CARD_COLLECTION]
    cards = pd.json_normalize(list(cards_collection.aggregate([])))

    GEMRATE_ID_COL = "GEMRATE_ID"
    METADATA_COLS = [
        GEMRATE_ID_COL,
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


# ==============================================================================
# PREPROCESSING PIPELINE
# ==============================================================================


def to_token_lists(X: np.ndarray) -> list[list[str]]:
    """Convert card numbers to token lists for hashing."""
    vals = pd.Series(np.ravel(X)).fillna("").astype(str)
    return [[f"card_number={v}"] for v in vals]


def get_preprocess_pipeline():
    """
    Build sklearn preprocessing pipeline for card metadata.

    Output dimension: ~1281
    - SVD-reduced categorical/hashed features: 256
    - Year (scaled): 1
    - Name embedding (bge-m3): 1024
    """
    CATEGORICAL_COLS = ["SET_NAME", "PARALLEL"]
    NUMERICAL_COLS = ["YEAR"]
    EMBEDDING_COLS = ["NAME"]
    HASHING_COLS = ["CARD_NUMBER"]

    # Sparse features: one-hot encoded categoricals + hashed card numbers
    sparse_block = ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=OHE_MIN_FREQUENCY,
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
                            FunctionTransformer(to_token_lists, validate=False),
                        ),
                        (
                            "hasher",
                            FeatureHasher(
                                n_features=HASH_N_FEATURES,
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

    # Reduce sparse features to dense with SVD
    sparse_to_dense = Pipeline(
        [
            ("sparse", sparse_block),
            (
                "svd",
                TruncatedSVD(n_components=SVD_N_COMPONENTS, random_state=RANDOM_SEED),
            ),
            (
                "to32",
                FunctionTransformer(lambda X: X.astype(np.float32), validate=False),
            ),
        ]
    )

    # Text embedding model for card names
    embed_model = SentenceTransformer("BAAI/bge-m3")

    def embed_text_batch(X):
        texts = pd.Series(np.ravel(X)).fillna("").astype(str).tolist()
        emb = embed_model.encode(texts, batch_size=64, show_progress_bar=True)
        return np.asarray(emb, dtype=np.float32)

    # Dense features: scaled year + text embeddings
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

    # Combine all features
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


# ==============================================================================
# SALES VECTOR BUILDER (Target Signals)
# ==============================================================================


def build_weekly_sales_vectors(sales_df):
    """
    Build weekly sales "fingerprints" for each card (aggregated across all grades).

    These vectors capture price behavior over time and serve as the
    learning target - cards with similar sales vectors should have
    similar embeddings.

    Returns:
        keys: DataFrame with gemrate_id
        S: Sales feature matrix [n_cards, n_features]
        obs_weeks: Number of observed weeks per card
    """
    print("Building Global-Time Sales Vectors...")
    s = sales_df.copy()
    s["date"] = pd.to_datetime(s["date"], utc=True)

    # Fixed start date for consistent week indexing
    global_start = pd.Timestamp("2018-01-01", tz="UTC")

    s["week"] = (s["date"] - global_start).dt.days // 7
    s = s.loc[s["week"] >= 0]
    n_weeks = s["week"].max()

    # Group by gemrate_id only (aggregate all grades together)
    group_cols = ["gemrate_id"]
    s["log_price"] = np.log1p(s["price"])

    # Pivot to get price and volume time series
    price_wide = s.pivot_table(
        index=group_cols, columns="week", values="log_price", aggfunc="median"
    )
    price_wide = price_wide.reindex(columns=range(n_weeks)).astype(np.float32)

    vol_wide = s.pivot_table(
        index=group_cols, columns="week", values="log_price", aggfunc="size"
    )
    vol_wide = vol_wide.reindex(columns=range(n_weeks)).fillna(0.0).astype(np.float32)

    # Z-normalize prices per card
    P = price_wide.to_numpy()
    M = (~np.isnan(P)).astype(np.float32)  # Observation mask
    P0 = np.nan_to_num(P, nan=0.0)

    denom = np.maximum(M.sum(axis=1, keepdims=True), 1.0)
    mu = (P0 * M).sum(axis=1, keepdims=True) / denom
    var = (((P0 - mu) * M) ** 2).sum(axis=1, keepdims=True) / denom
    sigma = np.sqrt(var).astype(np.float32)
    sigma = np.maximum(sigma, 1e-3)
    Pz = ((P0 - mu) / sigma) * M  # Z-normalized prices

    # Log-volume with cap
    V = np.minimum(vol_wide.to_numpy(), VOLUME_CAP_PER_WEEK)
    V_log = np.log1p(V)
    total_volume = V.sum(axis=1, keepdims=True)

    # Concatenate all features
    S = np.concatenate(
        [Pz, M, V_log, mu, sigma, np.log1p(total_volume)], axis=1
    ).astype(np.float32)

    keys = pd.DataFrame({"gemrate_id": price_wide.index.tolist()})

    return keys, S, M.sum(axis=1)


# ==============================================================================
# NEURAL NETWORK ARCHITECTURE
# ==============================================================================


class Encoder(nn.Module):
    """
    Main encoder network that maps card_metadata → embedding.

    Architecture:
        metadata (~1281d) → 2048 → 1024 → 768
        Final output is L2-normalized.
    """

    def __init__(self, meta_dim, out_dim=ENCODER_OUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(meta_dim, ENCODER_HIDDEN_1),
            nn.ReLU(),
            nn.Dropout(DROPOUT_P),
            nn.Linear(ENCODER_HIDDEN_1, ENCODER_HIDDEN_2),
            nn.ReLU(),
            nn.Dropout(DROPOUT_P),
            nn.Linear(ENCODER_HIDDEN_2, out_dim),
        )

    def forward(self, x_meta):
        z = self.net(x_meta)
        return F.normalize(z, p=2, dim=1)


# ==============================================================================
# CONTRASTIVE LOSS
# ==============================================================================


def cosine_sim_matrix(x):
    """Compute pairwise cosine similarity matrix."""
    x = F.normalize(x, p=2, dim=1)
    return x @ x.T


def compute_batch_loss(enc, xb_meta, sb, wb, tau):
    """
    Compute contrastive loss using KL divergence.

    The loss encourages the encoder's similarity matrix to match
    the sales vectors' similarity matrix.

    Args:
        enc: Encoder model
        xb_meta: Batch of metadata features
        sb: Batch of sales vectors (target)
        wb: Batch of observation weights
        tau: Temperature parameter

    Returns:
        Weighted KL divergence loss
    """
    w = torch.clamp(wb, min=0.0)
    w = w / (w.sum() + 1e-8)

    z = enc(xb_meta)
    Kz = cosine_sim_matrix(z)  # Encoder similarity
    Ks = cosine_sim_matrix(sb)  # Sales similarity (target)

    # Mask diagonal
    b = z.size(0)
    diag = torch.eye(b, device=z.device).bool()
    Kz = Kz.masked_fill(diag, -1e9)
    Ks = Ks.masked_fill(diag, -1e9)

    # KL divergence: encoder distribution vs sales distribution
    logP = F.log_softmax(Kz / tau, dim=1)
    Q = F.softmax(Ks / tau, dim=1)

    row_kl = F.kl_div(logP, Q, reduction="none").sum(dim=1)
    loss = (row_kl * w).sum() * (tau * tau)
    return loss


# ==============================================================================
# TRAINING FUNCTION
# ==============================================================================


def train_embeddings():
    """
    Main training function for the embedding model.

    Steps:
        1. Load card metadata from MongoDB
        2. Build preprocessing pipeline
        3. Build sales vectors from historical data
        4. Train encoder with contrastive loss
        5. Generate embeddings for all cards
        6. Save embeddings and model checkpoint
    """
    print("Starting Embedding Training...")

    # Device selection: CUDA -> MPS -> CPU
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Load card metadata
    cards = load_cards_from_mongo()
    if cards.empty:
        raise ValueError("No cards found in MongoDB.")

    # Process metadata features
    print("Processing metadata...")
    preprocess = get_preprocess_pipeline()
    X_all = preprocess.fit_transform(cards).astype(np.float32)
    X_all = np.nan_to_num(X_all)

    # Load sales data
    if not os.path.exists(FEATURES_PREPPED_FILE):
        raise FileNotFoundError(
            f"{FEATURES_PREPPED_FILE} not found. Run feature prep first."
        )

    sales_df = pd.read_csv(FEATURES_PREPPED_FILE)
    sales_df["gemrate_id"] = sales_df["gemrate_id"].astype(str)
    sales_keys, S, obs_weeks = build_weekly_sales_vectors(sales_df)

    # Align sales data with card metadata
    id_to_row = {str(gid): i for i, gid in enumerate(cards["GEMRATE_ID"])}
    meta_feats, S_aligned, obs_aligned, aligned_keys = [], [], [], []

    for i in range(len(sales_keys)):
        gid = str(sales_keys.iloc[i]["gemrate_id"])
        if gid in id_to_row:
            idx = id_to_row[gid]
            meta_feats.append(X_all[idx])
            S_aligned.append(S[i])
            obs_aligned.append(obs_weeks[i])
            aligned_keys.append(sales_keys.iloc[i])

    print(f"Aligned {len(meta_feats)} cards to sales data.")
    X_meta = np.stack(meta_feats)
    S_final = np.stack(S_aligned)
    W_final = np.array(obs_aligned).astype(np.float32)
    keys_df = pd.DataFrame(aligned_keys)

    # Train/val split (stratified by card ID)
    print("Performing Stratified Split by gemrate_id...")
    gss = GroupShuffleSplit(
        n_splits=1, test_size=FINETUNE_VAL_SPLIT, random_state=RANDOM_SEED
    )
    train_idx, val_idx = next(gss.split(X_meta, groups=keys_df["gemrate_id"]))
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)}")

    # Initialize model and optimizer
    enc = Encoder(meta_dim=X_meta.shape[1]).to(device)
    opt = torch.optim.AdamW(enc.parameters(), lr=FINETUNE_LR)

    # Create data loader
    train_dataset = TensorDataset(
        torch.from_numpy(X_meta[train_idx]),
        torch.from_numpy(S_final[train_idx]),
        torch.from_numpy(W_final[train_idx]),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=FINETUNE_BATCH_SIZE, shuffle=True, drop_last=True
    )

    # Training loop
    print(f"Starting Finetuning on {device}...")
    for epoch in range(FINETUNE_EPOCHS):
        enc.train()
        total_loss = 0.0

        for batch_meta, batch_s, batch_w in train_loader:
            b_meta = batch_meta.to(device)
            b_s = batch_s.to(device)
            b_w = batch_w.to(device)

            loss = compute_batch_loss(enc, b_meta, b_s, b_w, FINETUNE_TAU)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{FINETUNE_EPOCHS}: Avg Loss {avg_loss:.4f}")

    # Save model checkpoint
    torch.save(enc.state_dict(), MODEL_CHECKPOINT_FILE)
    print(f"Saved model checkpoint to {MODEL_CHECKPOINT_FILE}")

    # Generate embeddings for all cards (one per card)
    print("Generating static card embeddings...")
    enc.eval()
    final_embs = []
    final_keys = []

    with torch.no_grad():
        for i in range(0, len(X_all), EMBED_BATCH_SIZE):
            xm = torch.from_numpy(X_all[i : i + EMBED_BATCH_SIZE]).to(device)
            emb = enc(xm).cpu().numpy()
            final_embs.append(emb)

        for _, r in cards.iterrows():
            final_keys.append({"gemrate_id": str(r["GEMRATE_ID"])})

    # Save embeddings
    np.save(EMBEDDING_VECTORS_FILE, np.vstack(final_embs))
    pd.DataFrame(final_keys).to_csv(EMBEDDING_KEYS_FILE, index=False)
    print(f"Saved {len(final_keys)} embeddings to {EMBEDDING_VECTORS_FILE}")
    print("Embedding Training Complete.")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    train_embeddings()
