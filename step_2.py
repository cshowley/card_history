import os
import sys
import time

import pandas as pd
from sentence_transformers import SentenceTransformer

import constants
from data_integrity import get_tracker


def run_step_2():
    print("Starting Step 2: Text Embedding...")
    start_time = time.time()
    tracker = get_tracker()

    if not os.path.exists(constants.S2_INPUT_CATALOG_FILE):
        print(f"Error: Input file '{constants.S2_INPUT_CATALOG_FILE}' not found.")
        sys.exit(1)

    print(f"Loading data from {constants.S2_INPUT_CATALOG_FILE}...")
    df = pd.read_csv(constants.S2_INPUT_CATALOG_FILE, low_memory=False)

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

    print(f"Loading model: {constants.S2_MODEL_NAME}...")
    model = SentenceTransformer(constants.S2_MODEL_NAME, device=constants.DEVICE)

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

    print(f"Saving {len(output_df)} rows to {constants.S2_OUTPUT_EMBEDDINGS_FILE}...")
    output_df.to_parquet(constants.S2_OUTPUT_EMBEDDINGS_FILE, index=False)

    # Data Integrity Tracking
    duration = time.time() - start_time
    embedding_dim = embeddings.shape[1] if len(embeddings) > 0 else 0

    tracker.add_metric(
        id="s2_cards_embedded",
        title="Cards Embedded",
        value=f"{len(output_df):,}",
    )
    tracker.add_metric(
        id="s2_embedding_dim",
        title="Embedding Dimension",
        value=str(embedding_dim),
    )
    tracker.add_metric(
        id="s2_duration",
        title="Step 2 Duration",
        value=f"{duration:.1f}s",
    )

    print("Step 2 Complete.")
