# =========================
# Metadata-only embedding model with sales-history alignment (cold-start capable)
#
# GOAL:
# Learn f(metadata) → 768-d vector such that cards with similar price level AND
# price trajectory shape are close in embedding space. Cards without sales history
# still get embeddings inferred from metadata patterns.
#
# ARCHITECTURE:
# - Input: card metadata only (CATEGORY, YEAR, SET_NAME, NAME, PARALLEL, CARD_NUMBER)
# - Training signal: ~55k cards with sales history define "good neighbors"
# - Sales data is NEVER fed to encoder input; only used to define similarity targets
# - Output: 768-d L2-normalized vectors for vector DB nearest-neighbor retrieval
#
# STEPS:
# 1) metadata → dense feature vector X (sklearn + SentenceTransformer)
# 2) (optional) self-supervised metadata pretrain (denoising reconstruction)
# 3) fine-tune encoder so embedding neighbors match sales-history neighbors,
#    weighted by observed_weeks (sparse histories matter less)
# 4) embed ALL cards (including no-sales) into 768-d vectors for vector DB
# =========================

import os
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from pymongo import MongoClient

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

from sentence_transformers import SentenceTransformer

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# =========================
# HYPERPARAMETERS / CONSTANTS
# =========================
RANDOM_SEED = 42
N_WEEKS = 20
HASH_N_FEATURES = 2**12  # 4096; tune 2**10..2**14
SVD_N_COMPONENTS = 256  # tune 128/256/512
OHE_MIN_FREQUENCY = 10  # collapse categories with fewer samples
VOLUME_CAP_PER_WEEK = 200.0
WINSOR_Q_LOW = 0.001
WINSOR_Q_HIGH = 0.999

ENCODER_HIDDEN_1 = 2048
ENCODER_HIDDEN_2 = 1024
ENCODER_OUT_DIM = 768
DROPOUT_P = 0.1

PRETRAIN_EPOCHS = 5
PRETRAIN_BATCH_SIZE = 512
PRETRAIN_LR = 1e-3
PRETRAIN_WEIGHT_DECAY = 1e-4
PRETRAIN_NOISE_P = 0.1

FINETUNE_EPOCHS = 20
FINETUNE_BATCH_SIZE = 256
FINETUNE_LR = 5e-4
FINETUNE_WEIGHT_DECAY = 1e-4
FINETUNE_TAU = 0.07
FINETUNE_REG_LAMBDA = 0.05
FINETUNE_VAL_SPLIT = 0.1

EMBED_BATCH_SIZE = 4096


# =========================
# DEVICE SELECTION (CUDA > MPS > CPU)
# =========================
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


device = get_device()
print(f"Using device: {device}")


# =========================
# 0) LOAD METADATA (CARDS)
# =========================
print("Connecting to MongoDB...")
load_dotenv()
client = MongoClient(os.environ["MONGO_URI"])
db = client["gemrate"]

print("Loading cards from MongoDB (this may take a minute)...")
cards_collection = db["gemrate_pokemon_cards"]
cards = pd.json_normalize(list(cards_collection.aggregate([])))
print(f"  → Loaded {len(cards)} raw records from MongoDB")

SPEC_ID_COL = "SPECID"
METADATA_COLS = [
    SPEC_ID_COL,
    "CATEGORY",
    "YEAR",
    "SET_NAME",
    "NAME",
    "PARALLEL",
    "CARD_NUMBER",
]

missing = [c for c in METADATA_COLS if c not in cards.columns]
if missing:
    raise KeyError(f"Metadata missing columns: {missing}")

cards = cards[METADATA_COLS].copy()
cards[SPEC_ID_COL] = cards[SPEC_ID_COL].astype(str)
cards["YEAR"] = pd.to_numeric(cards["YEAR"], errors="coerce")

if not cards[SPEC_ID_COL].is_unique:
    raise ValueError(f"cards['{SPEC_ID_COL}'] must be unique (one row per spec_id).")

print(f"Loaded {len(cards)} cards")


# =========================
# 1) PREPROCESS PIPELINE: METADATA → DENSE FLOAT32 VECTOR
# =========================
CATEGORICAL_COLS = ["CATEGORY", "SET_NAME", "PARALLEL"]
NUMERICAL_COLS = ["YEAR"]
EMBEDDING_COLS = ["NAME"]
HASHING_COLS = ["CARD_NUMBER"]


def to_token_lists(X: np.ndarray) -> list[list[str]]:
    vals = pd.Series(np.ravel(X)).fillna("").astype(str)
    return [[f"card_number={v}"] for v in vals]


sparse_block = ColumnTransformer(
    transformers=[
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
                steps=[
                    ("to_tokens", FunctionTransformer(to_token_lists, validate=False)),
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

sparse_to_dense = Pipeline(
    steps=[
        ("sparse", sparse_block),
        ("svd", TruncatedSVD(n_components=SVD_N_COMPONENTS, random_state=RANDOM_SEED)),
        ("to32", FunctionTransformer(lambda X: X.astype(np.float32), validate=False)),
    ]
)

# Using bge-m3 - stable, well-tested model with 1024-dim output
embed_model = SentenceTransformer("BAAI/bge-m3")


def embed_text_batch(X: np.ndarray) -> np.ndarray:
    texts = pd.Series(np.ravel(X)).fillna("").astype(str).tolist()
    # NOTE: Do NOT use 'prompt' parameter - it causes NaN outputs with this model
    emb = embed_model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
    )
    return np.asarray(emb, dtype=np.float32)


dense_block = ColumnTransformer(
    transformers=[
        (
            "year",
            Pipeline(
                steps=[
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
        ("name", FunctionTransformer(embed_text_batch, validate=False), EMBEDDING_COLS),
    ],
    remainder="drop",
)

preprocess = Pipeline(
    steps=[
        (
            "union",
            FeatureUnion(
                [
                    ("meta_lowdim", sparse_to_dense),
                    ("dense", dense_block),
                ]
            ),
        ),
    ]
)

print("Fitting preprocess pipeline...")
X_all = preprocess.fit_transform(cards).astype(np.float32)
print(f"X_all shape: {X_all.shape}")

# Check for NaN/inf and replace with 0
nan_count = np.isnan(X_all).sum()
inf_count = np.isinf(X_all).sum()
if nan_count > 0 or inf_count > 0:
    print(
        f"  ⚠️ Found {nan_count} NaN and {inf_count} inf values in X_all, replacing with 0"
    )
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)


# =========================
# 2) LOAD / CLEAN SALES DATA
# =========================
def clean_sales_df(sales_df: pd.DataFrame) -> pd.DataFrame:
    s = sales_df.copy()
    required = ["spec_id", "date", "price"]
    for c in required:
        if c not in s.columns:
            raise KeyError(f"sales_df missing required column: {c}")
    s["spec_id"] = s["spec_id"].astype(str)
    s["date"] = pd.to_datetime(s["date"], utc=True, errors="coerce")
    s["price"] = pd.to_numeric(s["price"], errors="coerce")
    s = s.dropna(subset=required)
    return s


# =========================
# 3) BUILD WEEKLY SALES VECTORS (TARGETS ONLY; ANCHORED PER SPEC_ID)
# =========================
def build_weekly_sales_vectors_per_spec(
    sales_df: pd.DataFrame,
    n_weeks: int = N_WEEKS,
    id_col: str = "spec_id",
    date_col: str = "date",
    price_col: str = "price",
    volume_cap_per_week: float = VOLUME_CAP_PER_WEEK,
    winsor_q_low: float | None = WINSOR_Q_LOW,
    winsor_q_high: float | None = WINSOR_Q_HIGH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        sales_ids: array of spec_id values
        S: sales target vectors (n_sales, feature_dim)
        obs_weeks: observed weeks per card (n_sales,)
    """
    s = sales_df[[id_col, date_col, price_col]].copy()
    s[id_col] = s[id_col].astype(str)
    s[date_col] = pd.to_datetime(s[date_col], utc=True, errors="coerce")
    s[price_col] = pd.to_numeric(s[price_col], errors="coerce")
    s = s.dropna(subset=[id_col, date_col, price_col])

    if winsor_q_low is not None and winsor_q_high is not None:
        lo = s[price_col].quantile(winsor_q_low)
        hi = s[price_col].quantile(winsor_q_high)
        s[price_col] = s[price_col].clip(lo, hi)

    # anchor weeks per spec_id (each card's last sale = week 0)
    end_per_id = s.groupby(id_col)[date_col].transform("max")
    s["week"] = ((end_per_id - s[date_col]).dt.days // 7).astype(int)
    s = s[(s["week"] >= 0) & (s["week"] < n_weeks)]

    s["log_price"] = np.log1p(s[price_col].astype(float))

    # weekly median log-price (absolute level trajectory)
    price_wide = (
        s.pivot_table(
            index=id_col, columns="week", values="log_price", aggfunc="median"
        )
        .reindex(columns=range(n_weeks))
        .astype(np.float32)
    )

    # weekly volume (count of sales events)
    vol_wide = (
        s.pivot_table(index=id_col, columns="week", values="log_price", aggfunc="size")
        .reindex(columns=range(n_weeks))
        .fillna(0.0)
        .astype(np.float32)
    )

    P = price_wide.to_numpy()
    M = (~np.isnan(P)).astype(np.float32)
    P0 = np.nan_to_num(P, nan=0.0).astype(np.float32)

    # shape (z-score across observed weeks)
    denom = np.maximum(M.sum(axis=1, keepdims=True), 1.0)
    mu = (P0 * M).sum(axis=1, keepdims=True) / denom
    var = (((P0 - mu) * M) ** 2).sum(axis=1, keepdims=True) / denom
    sigma = np.sqrt(var).astype(np.float32)
    sigma = np.maximum(sigma, 1e-3)
    Pz = ((P0 - mu) / sigma) * M

    V = vol_wide.to_numpy().astype(np.float32)
    V = np.minimum(V, volume_cap_per_week).astype(np.float32)
    V_log = np.log1p(V)

    total_volume = V.sum(axis=1, keepdims=True).astype(np.float32)
    observed_weeks = M.sum(axis=1, keepdims=True).astype(np.float32)

    mean_log_price = mu.astype(np.float32)
    std_log_price = sigma.astype(np.float32)

    S = np.concatenate(
        [P0, Pz, M, V_log, mean_log_price, std_log_price, np.log1p(total_volume)],
        axis=1,
    ).astype(np.float32)

    sales_ids = price_wide.index.to_numpy()
    obs_weeks_flat = observed_weeks.ravel().astype(np.float32)

    return sales_ids, S, obs_weeks_flat


def align_sales_to_cards(
    cards_df: pd.DataFrame,
    X_all: np.ndarray,
    sales_ids: np.ndarray,
    S: np.ndarray,
    obs_weeks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        rows: indices into cards_df / X_all
        X_sales: metadata features for sales subset
        S_aligned: sales target vectors aligned to rows
        obs_weeks_aligned: observed weeks aligned to rows
    """
    id_to_row = {
        sid: i for i, sid in enumerate(cards_df[SPEC_ID_COL].astype(str).tolist())
    }

    rows = []
    S_rows = []
    obs_rows = []
    for sid, vec, ow in zip(sales_ids, S, obs_weeks):
        sid = str(sid)
        if sid in id_to_row:
            rows.append(id_to_row[sid])
            S_rows.append(vec)
            obs_rows.append(ow)

    if not rows:
        raise ValueError(
            "No overlap between cards.spec_id and sales.spec_id after alignment."
        )

    rows = np.array(rows, dtype=np.int64)
    S_aligned = np.stack(S_rows).astype(np.float32)
    obs_weeks_aligned = np.array(obs_rows, dtype=np.float32)
    X_sales = X_all[rows].astype(np.float32)

    return rows, X_sales, S_aligned, obs_weeks_aligned


# =========================
# 4) ENCODER / DECODER MODELS
# =========================
class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_1: int = ENCODER_HIDDEN_1,
        hidden_2: int = ENCODER_HIDDEN_2,
        out_dim: int = ENCODER_OUT_DIM,
        dropout_p: float = DROPOUT_P,
        normalize_output: bool = True,
    ):
        super().__init__()
        self.normalize_output = normalize_output
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_1),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        if self.normalize_output:
            z = F.normalize(z, p=2, dim=1)
        return z


class Decoder(nn.Module):
    def __init__(
        self,
        out_dim: int,
        hidden_2: int = ENCODER_HIDDEN_2,
        hidden_1: int = ENCODER_HIDDEN_1,
        input_dim: int = None,
        dropout_p: float = DROPOUT_P,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(out_dim, hidden_2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_2, hidden_1),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_1, input_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# =========================
# 5) UTILITY FUNCTIONS
# =========================
def corrupt(x: torch.Tensor, p: float = 0.1) -> torch.Tensor:
    mask = (torch.rand_like(x) > p).float()
    return x * mask


def cosine_sim_matrix(x: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x, p=2, dim=1)
    return x @ x.T


@torch.no_grad()
def snapshot_encoder_outputs(
    enc: nn.Module,
    X: np.ndarray,
    batch_size: int = EMBED_BATCH_SIZE,
) -> np.ndarray:
    enc.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i : i + batch_size]).to(device)
        out.append(enc(xb).cpu().numpy())
    return np.vstack(out).astype(np.float32)


# =========================
# 6) PRETRAIN: DENOISING RECONSTRUCTION (OPTIONAL)
# =========================
def pretrain_denoising(
    enc: nn.Module,
    dec: nn.Module,
    X_all: np.ndarray,
    epochs: int = PRETRAIN_EPOCHS,
    batch_size: int = PRETRAIN_BATCH_SIZE,
    lr: float = PRETRAIN_LR,
    weight_decay: float = PRETRAIN_WEIGHT_DECAY,
    noise_p: float = PRETRAIN_NOISE_P,
) -> None:
    # temporarily disable normalization for pretrain (optional; remove if you prefer normalized)
    original_normalize = enc.normalize_output
    enc.normalize_output = False

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_all)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    opt = torch.optim.AdamW(
        list(enc.parameters()) + list(dec.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    for epoch in range(1, epochs + 1):
        enc.train()
        dec.train()
        total = 0.0

        for (xb,) in loader:
            xb = xb.to(device)
            z = enc(corrupt(xb, p=noise_p))
            x_hat = dec(z)
            loss = F.mse_loss(x_hat, xb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        print(f"pretrain epoch={epoch:02d} mse={total / len(loader):.6f}")

    # restore normalization setting
    enc.normalize_output = original_normalize
    print("Pretrain complete.")


# =========================
# 7) FINETUNE: SALES-ALIGNMENT WITH OBSERVED_WEEKS WEIGHTING + VALIDATION
# =========================
class IndexedDatasetWithWeights(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, S: np.ndarray, W: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.S = torch.from_numpy(S.astype(np.float32))
        self.W = torch.from_numpy(W.astype(np.float32))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        return self.X[idx], self.S[idx], self.W[idx], idx


def compute_batch_loss(
    enc: nn.Module,
    xb: torch.Tensor,
    sb: torch.Tensor,
    wb: torch.Tensor,
    z0: torch.Tensor | None,
    tau: float,
    reg_lambda: float,
) -> torch.Tensor:
    """Compute weighted KL alignment loss + optional regularization."""
    # normalize weights to sum to 1
    w = torch.clamp(wb, min=0.0)
    w = w / (w.sum() + 1e-8)

    z = enc(xb)
    Kz = cosine_sim_matrix(z)
    Ks = cosine_sim_matrix(sb)

    b = xb.size(0)
    diag = torch.eye(b, device=xb.device).bool()
    Kz = Kz.masked_fill(diag, -1e9)
    Ks = Ks.masked_fill(diag, -1e9)

    logP = F.log_softmax(Kz / tau, dim=1)
    Q = F.softmax(Ks / tau, dim=1)

    row_kl = F.kl_div(logP, Q, reduction="none").sum(dim=1)
    loss_align = (row_kl * w).sum() * (tau * tau)

    loss = loss_align

    if z0 is not None and reg_lambda > 0:
        row_reg = ((z - z0) ** 2).mean(dim=1)
        loss_reg = (row_reg * w).sum()
        loss = loss + reg_lambda * loss_reg

    return loss


@torch.no_grad()
def evaluate_validation_loss(
    enc: nn.Module,
    X_val: np.ndarray,
    S_val: np.ndarray,
    obs_weeks_val: np.ndarray,
    tau: float,
    batch_size: int,
) -> float:
    """Compute validation loss (no regularization)."""
    enc.eval()
    ds = IndexedDatasetWithWeights(X_val, S_val, obs_weeks_val)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

    total = 0.0
    count = 0

    for xb, sb, wb, _ in loader:
        xb = xb.to(device)
        sb = sb.to(device)
        wb = wb.to(device)

        w = torch.clamp(wb, min=0.0)
        w = w / (w.sum() + 1e-8)

        z = enc(xb)
        Kz = cosine_sim_matrix(z)
        Ks = cosine_sim_matrix(sb)

        b = xb.size(0)
        if b < 2:
            continue

        diag = torch.eye(b, device=device).bool()
        Kz = Kz.masked_fill(diag, -1e9)
        Ks = Ks.masked_fill(diag, -1e9)

        logP = F.log_softmax(Kz / tau, dim=1)
        Q = F.softmax(Ks / tau, dim=1)

        row_kl = F.kl_div(logP, Q, reduction="none").sum(dim=1)
        loss = (row_kl * w).sum() * (tau * tau)

        total += loss.item()
        count += 1

    return total / max(count, 1)


def finetune_sales_alignment_with_reg_weighted(
    enc: nn.Module,
    X_sales: np.ndarray,
    S_aligned: np.ndarray,
    obs_weeks: np.ndarray,
    epochs: int = FINETUNE_EPOCHS,
    batch_size: int = FINETUNE_BATCH_SIZE,
    lr: float = FINETUNE_LR,
    weight_decay: float = FINETUNE_WEIGHT_DECAY,
    tau: float = FINETUNE_TAU,
    reg_lambda: float = FINETUNE_REG_LAMBDA,
    val_split: float = FINETUNE_VAL_SPLIT,
    checkpoint_path: str = "best_encoder.pt",
) -> nn.Module:
    """
    Fine-tune encoder with sales-alignment loss, weighted by observed_weeks.
    Includes train/val split, early stopping, and checkpointing.
    """
    # train/val split
    n = len(X_sales)
    indices = np.arange(n)
    idx_train, idx_val = train_test_split(
        indices, test_size=val_split, random_state=RANDOM_SEED
    )

    X_train, X_val = X_sales[idx_train], X_sales[idx_val]
    S_train, S_val = S_aligned[idx_train], S_aligned[idx_val]
    obs_train, obs_val = obs_weeks[idx_train], obs_weeks[idx_val]

    print(f"Finetune split: {len(idx_train)} train, {len(idx_val)} val")

    # snapshot pre-finetune embeddings for regularization
    Z0_full = snapshot_encoder_outputs(enc, X_sales)
    Z0_train = Z0_full[idx_train]

    ds_train = IndexedDatasetWithWeights(X_train, S_train, obs_train)
    loader_train = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True, drop_last=True
    )

    opt = torch.optim.AdamW(enc.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        enc.train()
        total_train = 0.0

        for xb, sb, wb, idx in loader_train:
            xb = xb.to(device)
            sb = sb.to(device)
            wb = wb.to(device)
            idx_np = idx.numpy()

            z0 = torch.from_numpy(Z0_train[idx_np]).to(device)

            loss = compute_batch_loss(enc, xb, sb, wb, z0, tau, reg_lambda)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_train += loss.item()

        avg_train = total_train / len(loader_train)
        avg_val = evaluate_validation_loss(enc, X_val, S_val, obs_val, tau, batch_size)

        print(
            f"finetune epoch={epoch:02d} train_loss={avg_train:.6f} val_loss={avg_val:.6f}"
        )

        # checkpointing
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(enc.state_dict(), checkpoint_path)
            print(f"  → saved checkpoint (val_loss={avg_val:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)"
                )
                break

    # load best checkpoint (if it exists)
    if os.path.exists(checkpoint_path):
        enc.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded best checkpoint with val_loss={best_val_loss:.6f}")
    else:
        print(
            "⚠️ No checkpoint was saved (all losses were NaN). Using current model state."
        )

    return enc


# =========================
# 8) EMBED ALL CARDS FOR VECTOR DB
# =========================
@torch.no_grad()
def embed_all_cards(
    enc: nn.Module,
    preprocess_pipeline: Pipeline,
    cards_df: pd.DataFrame,
    batch_size: int = EMBED_BATCH_SIZE,
) -> np.ndarray:
    enc.eval()
    X = preprocess_pipeline.transform(cards_df).astype(np.float32)
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i : i + batch_size]).to(device)
        out.append(enc(xb).cpu().numpy())
    return np.vstack(out).astype(np.float32)


# =========================
# 9) EVALUATION: CHECK IF EMBEDDINGS CAPTURE SALES SIMILARITY
# =========================
@torch.no_grad()
def evaluate_embedding_quality(
    enc: nn.Module,
    X_eval: np.ndarray,
    S_eval: np.ndarray,
    k: int = 10,
) -> dict:
    """
    For each card in eval set, find k nearest neighbors by embedding,
    then compute correlation between embedding similarity and sales similarity.
    """
    enc.eval()
    Z = snapshot_encoder_outputs(enc, X_eval)

    # cosine similarity matrices
    Z_norm = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
    S_norm = S_eval / (np.linalg.norm(S_eval, axis=1, keepdims=True) + 1e-8)

    sim_Z = Z_norm @ Z_norm.T
    sim_S = S_norm @ S_norm.T

    n = len(Z)
    correlations = []

    for i in range(n):
        # exclude self
        mask = np.ones(n, dtype=bool)
        mask[i] = False

        z_sims = sim_Z[i, mask]
        s_sims = sim_S[i, mask]

        if np.std(z_sims) > 1e-8 and np.std(s_sims) > 1e-8:
            corr = np.corrcoef(z_sims, s_sims)[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)

    return {
        "mean_correlation": np.mean(correlations),
        "median_correlation": np.median(correlations),
        "std_correlation": np.std(correlations),
        "n_evaluated": len(correlations),
    }


# =========================
# 10) MAIN EXECUTION
# =========================
if __name__ == "__main__":
    # --------------------------------------------------
    # LOAD SALES DATA FROM CSV
    # --------------------------------------------------
    SALES_CSV_PATH = os.path.join(os.path.dirname(__file__), "full_training_data.csv")
    print(f"Loading sales data from {SALES_CSV_PATH}...")
    sales_df = pd.read_csv(SALES_CSV_PATH)
    sales_df = clean_sales_df(sales_df)
    print(f"Loaded {len(sales_df)} sales records")

    # --------------------------------------------------
    # BUILD SALES VECTORS
    # --------------------------------------------------
    sales_ids, S, obs_weeks = build_weekly_sales_vectors_per_spec(
        sales_df, n_weeks=N_WEEKS
    )
    rows, X_sales, S_aligned, obs_weeks_aligned = align_sales_to_cards(
        cards, X_all, sales_ids, S, obs_weeks
    )
    print(f"Sales-supervised subset: X={X_sales.shape}, S={S_aligned.shape}")

    # --------------------------------------------------
    # INITIALIZE MODELS
    # --------------------------------------------------
    enc = Encoder(input_dim=X_all.shape[1], out_dim=ENCODER_OUT_DIM).to(device)
    dec = Decoder(out_dim=ENCODER_OUT_DIM, input_dim=X_all.shape[1]).to(device)

    # --------------------------------------------------
    # STEP 1: PRETRAIN (OPTIONAL BUT RECOMMENDED)
    # --------------------------------------------------
    pretrain_denoising(
        enc, dec, X_all, epochs=PRETRAIN_EPOCHS, noise_p=PRETRAIN_NOISE_P
    )

    # --------------------------------------------------
    # STEP 2: FINETUNE WITH SALES ALIGNMENT
    # --------------------------------------------------
    enc = finetune_sales_alignment_with_reg_weighted(
        enc,
        X_sales,
        S_aligned,
        obs_weeks_aligned,
        epochs=FINETUNE_EPOCHS,
        batch_size=FINETUNE_BATCH_SIZE,
        reg_lambda=FINETUNE_REG_LAMBDA,
        tau=FINETUNE_TAU,
        checkpoint_path="best_encoder.pt",
    )

    # --------------------------------------------------
    # STEP 3: EVALUATE
    # --------------------------------------------------
    eval_results = evaluate_embedding_quality(enc, X_sales, S_aligned, k=10)
    print(f"Embedding quality: {eval_results}")

    # --------------------------------------------------
    # STEP 4: EMBED ALL CARDS
    # --------------------------------------------------
    Z_all_768 = embed_all_cards(enc, preprocess, cards)
    print(f"Z_all_768 shape: {Z_all_768.shape}")

    # --------------------------------------------------
    # STEP 5: SAVE
    # --------------------------------------------------
    np.save("card_vectors_768.npy", Z_all_768)
    np.save("spec_ids.npy", cards["SPECID"].values)
    np.save("sales_spec_ids.npy", sales_ids)  # spec_ids that have sales history
    torch.save(enc.state_dict(), "final_encoder.pt")
    print("Saved embeddings, spec_ids, sales_spec_ids, and model.")

    print("Training complete!")
