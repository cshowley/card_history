"""
Step 5: Neighbor Search

Uses the unified 768-dim embeddings from step 2 to find similar cards.
For each card, finds the N most similar cards based on cosine similarity.
"""

import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import constants


def s5_load_embeddings():
    """Load unified embeddings and prepare for search."""
    print(f"Loading embeddings from {constants.S3_OUTPUT_EMBEDDINGS_FILE}...")
    emb_df = pd.read_parquet(constants.S3_OUTPUT_EMBEDDINGS_FILE)
    
    if "gemrate_id" in emb_df.columns:
        emb_df = emb_df.set_index("gemrate_id")
    
    # Stack embedding vectors into matrix
    matrix = np.array(emb_df["embedding_vector"].tolist())
    
    # L2 normalize for cosine similarity
    norm = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    matrix_norm = matrix / norm
    
    embeddings_norm = pd.DataFrame(matrix_norm, index=emb_df.index)
    
    print(f"Loaded {len(embeddings_norm)} embeddings with dimension {matrix_norm.shape[1]}")
    
    return embeddings_norm


def run_step_5():
    """
    Find nearest neighbors for all cards using unified embeddings.
    
    Uses cosine similarity to find the most similar cards.
    """
    print("Starting Step 5: Neighbor Search...")
    
    if not os.path.exists(constants.S3_OUTPUT_EMBEDDINGS_FILE):
        print(f"Embeddings file {constants.S3_OUTPUT_EMBEDDINGS_FILE} not found.")
        print("Run step 3 first to generate embeddings.")
        return
    
    # Load embeddings
    embeddings_norm = s5_load_embeddings()
    
    # Create tensors for GPU search
    db_ids = embeddings_norm.index.sort_values()
    db_tensor = torch.tensor(
        embeddings_norm.loc[db_ids].values,
        dtype=torch.float32,
        device=constants.DEVICE
    )
    
    query_ids = embeddings_norm.index
    query_tensor = torch.tensor(
        embeddings_norm.loc[query_ids].values,
        dtype=torch.float32,
        device=constants.DEVICE
    )
    
    print(f"Database: {len(db_ids)} cards")
    print(f"Query set: {len(query_ids)} cards")
    print(f"Using device: {constants.DEVICE}")
    
    # Compute similarity in batches
    batch_size = 4096
    num_queries = len(query_ids)
    
    all_gemrate_ids = []
    all_neighbors = []
    all_scores = []
    
    for i in tqdm(range(0, num_queries, batch_size), desc="Searching"):
        end_idx = min(i + batch_size, num_queries)
        batch_query = query_tensor[i:end_idx]
        
        with torch.no_grad():
            # Cosine similarity (vectors are already normalized)
            sim = torch.matmul(batch_query, db_tensor.T)
            
            # Get top-k (including self, which we'll filter out)
            k_val = min(constants.S5_N_NEIGHBORS_PREPARE + 1, sim.shape[1])
            top_vals, top_idxs = torch.topk(sim, k=k_val, dim=1)
            
            top_idxs = top_idxs.cpu().numpy()
            top_vals = top_vals.cpu().numpy()
        
        batch_ids = query_ids[i:end_idx]
        
        for local_idx, query_id in enumerate(batch_ids):
            indices = top_idxs[local_idx]
            vals = top_vals[local_idx]
            
            neighbor_ids = db_ids[indices]
            
            # Filter out self-matches
            mask = neighbor_ids != query_id
            final_ids = neighbor_ids[mask][:constants.S5_N_NEIGHBORS_PREPARE]
            final_scores = vals[mask][:constants.S5_N_NEIGHBORS_PREPARE]
            
            count = len(final_ids)
            if count > 0:
                all_gemrate_ids.extend([query_id] * count)
                all_neighbors.extend(final_ids)
                all_scores.extend(final_scores)
    
    # Create output dataframe
    results_df = pd.DataFrame({
        "gemrate_id": all_gemrate_ids,
        "neighbors": all_neighbors,
        "score": all_scores
    })
    
    print(f"Saving {len(results_df)} neighbor pairs to {constants.S5_OUTPUT_NEIGHBORS_FILE}...")
    print(f"Unique query IDs: {results_df['gemrate_id'].nunique()} / {num_queries}")
    
    results_df.to_parquet(constants.S5_OUTPUT_NEIGHBORS_FILE, index=False)
    print("Step 5 Complete.")
