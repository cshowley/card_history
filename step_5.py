import os
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import constants
from data_integrity import get_tracker


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

    db_text_tensor = torch.tensor(
        db_text_np, dtype=torch.float32, device=constants.DEVICE
    )
    db_price_tensor = torch.tensor(
        db_price_np, dtype=torch.float32, device=constants.DEVICE
    )

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
        text_emb_df_norm.loc[db_ids].values,
        dtype=torch.float32,
        device=constants.DEVICE,
    )
    db_price_tensor = torch.tensor(
        price_emb_df_norm.loc[db_ids].values,
        dtype=torch.float32,
        device=constants.DEVICE,
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
    start_time = time.time()
    tracker = get_tracker()

    if not os.path.exists(constants.S2_OUTPUT_EMBEDDINGS_FILE):
        print(f"Embeddings file {constants.S2_OUTPUT_EMBEDDINGS_FILE} not found.")
        return

    if not os.path.exists(constants.S4_OUTPUT_PRICE_VECS_FILE):
        print(f"Price vectors file {constants.S4_OUTPUT_PRICE_VECS_FILE} not found.")
        return

    price_vecs = pd.read_parquet(constants.S4_OUTPUT_PRICE_VECS_FILE)
    if "gemrate_id" in price_vecs.columns:
        price_vecs = price_vecs.set_index("gemrate_id")

    (
        db_text_tensor,
        db_price_tensor,
        db_ids,
        all_text_norm,
        all_price_norm,
        query_ids,
    ) = s5_prepare_matrices(price_vecs, constants.S2_OUTPUT_EMBEDDINGS_FILE)

    q_text_np = all_text_norm.loc[query_ids].values
    q_text = torch.tensor(q_text_np, dtype=torch.float32, device=constants.DEVICE)

    q_price_aligned = all_price_norm.reindex(query_ids, fill_value=0.0)
    q_price = torch.tensor(
        q_price_aligned.values, dtype=torch.float32, device=constants.DEVICE
    )

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

            k_val = min(constants.S5_N_NEIGHBORS_PREPARE + 1, total_sim.shape[1])
            top_vals, top_idxs = torch.topk(total_sim, k=k_val, dim=1)

            top_idxs = top_idxs.cpu().numpy()
            top_vals = top_vals.cpu().numpy()

        batch_ids = query_ids[i:end_idx]

        for local_idx, query_id in enumerate(batch_ids):
            indices = top_idxs[local_idx]
            vals = top_vals[local_idx]

            neighbor_ids = db_ids[indices]

            mask = neighbor_ids != query_id
            final_ids = neighbor_ids[mask][: constants.S5_N_NEIGHBORS_PREPARE]
            final_scores = vals[mask][: constants.S5_N_NEIGHBORS_PREPARE]

            count = len(final_ids)
            if count > 0:
                all_gemrate_ids.extend([query_id] * count)
                all_neighbors.extend(final_ids)
                all_scores.extend(final_scores)

    results_df = pd.DataFrame(
        {"gemrate_id": all_gemrate_ids, "neighbors": all_neighbors, "score": all_scores}
    )

    print(
        f"Saving final output with {len(results_df)} rows to {constants.S5_OUTPUT_NEIGHBORS_FILE}..."
    )
    print(
        f"Unique Query IDs processed: {results_df['gemrate_id'].nunique()} / {num_queries}"
    )
    results_df.to_parquet(constants.S5_OUTPUT_NEIGHBORS_FILE)

    # Data Integrity Tracking
    duration = time.time() - start_time
    unique_queries = results_df["gemrate_id"].nunique()
    total_catalog_cards = num_queries
    coverage_pct = round((unique_queries / total_catalog_cards) * 100, 2) if total_catalog_cards > 0 else 0.0

    tracker.add_metric(
        id="s5_total_catalog_cards",
        title="Total Catalog Cards (Query Set)",
        value=total_catalog_cards,
    )
    tracker.add_metric(
        id="s5_cards_with_neighbors",
        title="Cards with Neighbors",
        value=unique_queries,
    )
    tracker.add_metric(
        id="s5_catalog_coverage_pct",
        title="Catalog Coverage (%)",
        value=coverage_pct,
    )
    tracker.add_metric(
        id="s5_total_neighbor_pairs",
        title="Total Neighbor Pairs",
        value=len(results_df),
    )
    tracker.add_metric(
        id="s5_db_size",
        title="Database Size (Cards with Text + Price)",
        value=len(db_ids),
    )
    tracker.add_metric(
        id="s5_duration",
        title="Step 5 Duration",
        value=round(duration, 1),
    )

    print("Step 5 Complete.")
