import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
from tqdm import tqdm

SALES_DATA_FILE = 'features_prepped.csv'
EMBEDDINGS_FILE = 'gemrate_embeddings.parquet'
WINDOW_SIZE = 4
BATCH_SIZE = 64
EPOCHS = 10
PRICE_EMBEDDING_DIM = 32
LEARNING_RATE = 0.0001
N_NEIGHBORS = 5
OUTPUT_FILE = 'neighbors.parquet'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def normalize_grades(df):
    print("Normalizing grades...")
    grade_counts = df.groupby(['gemrate_id', 'grade']).size().reset_index(name='count')
    base_grades = grade_counts.sort_values(['gemrate_id', 'count'], ascending=[True, False]) \
                              .drop_duplicates('gemrate_id') \
                              .set_index('gemrate_id')['grade']
    
    avg_prices = df.groupby(['gemrate_id', 'grade'])['price'].mean()
    
    multipliers = avg_prices.reset_index(name='avg_price')
    multipliers['base_grade'] = multipliers['gemrate_id'].map(base_grades)
    
    base_prices = avg_prices.reset_index()
    base_prices = base_prices[base_prices['grade'] == base_prices['gemrate_id'].map(base_grades)]
    base_prices = base_prices.set_index('gemrate_id')['price']
    
    multipliers['base_price'] = multipliers['gemrate_id'].map(base_prices)
    multipliers['multiplier'] = multipliers['avg_price'] / multipliers['base_price']
    
    df = df.merge(multipliers[['gemrate_id', 'grade', 'multiplier']], on=['gemrate_id', 'grade'], how='left')
    
    df['normalized_price'] = df['price'] / df['multiplier']
    
    return df

def prepare_price_matrix(df):
    print("Preparing price matrix...")
    df['date'] = pd.to_datetime(df['date'], format='mixed')
    
    pivot_df = df.groupby(['date', 'gemrate_id'])['normalized_price'].mean().unstack()
    
    pivot_df = pivot_df.resample('W').mean()
    pivot_df = pivot_df.ffill(limit=4)
    
    pivot_df = pivot_df.dropna(axis=1)
    
    print(f"Price Matrix Shape after cleaning: {pivot_df.shape}")
    
    if pivot_df.empty:
        print("Warning: Price matrix is empty after dropna/ffill.")
        return pivot_df, [], []

    log_returns = np.log(pivot_df / pivot_df.shift(1)).dropna()
    
    data_values = log_returns.values
    card_ids = log_returns.columns
    
    if data_values.shape[0] < WINDOW_SIZE:
        print("Not enough time points for windowing.")
        return pivot_df, [], []

    print("Generating sliding windows...")
    
    raw_windows = []
    for i in range(len(data_values) - WINDOW_SIZE + 1):
        window = data_values[i : i + WINDOW_SIZE, :]
        raw_windows.append(window)
        
    raw_windows = np.array(raw_windows)
    
    processed = np.transpose(raw_windows, (2, 0, 1)) 
    
    processed = processed.reshape(-1, WINDOW_SIZE)
    
    means = processed.mean(axis=1, keepdims=True)
    stds = processed.std(axis=1, keepdims=True) + 1e-8
    processed = (processed - means) / stds
    
    num_time_steps = raw_windows.shape[0]
    ids_map = np.repeat(card_ids.values, num_time_steps)
    
    X = processed[:, :, np.newaxis]
    
    return pivot_df, X, ids_map

class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len=WINDOW_SIZE, n_features=1, embedding_dim=PRICE_EMBEDDING_DIM):
        super(LSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        
        self.encoder_lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=embedding_dim,
            batch_first=True
        )
        
        self.decoder_lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            batch_first=True
        )
        
        self.output_layer = nn.Linear(embedding_dim, n_features)

    def forward(self, x):
        _, (h_n, _) = self.encoder_lstm(x)
        embedding = h_n.squeeze(0)
        
        repeat_embedding = embedding.unsqueeze(1).repeat(1, self.seq_len, 1)
        
        decoder_output, _ = self.decoder_lstm(repeat_embedding)
        
        x_recon = self.output_layer(decoder_output)
        
        return x_recon, embedding

def train_and_extract(X_data, ids_map):
    print("Starting Training Process...")
    
    indices = np.arange(len(X_data))
    np.random.shuffle(indices)
    split_idx = int(len(X_data) * 0.8)
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    
    X_train = X_data[train_idx]
    X_val = X_data[val_idx]
    
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32))
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    print("Phase 1: Finding optimal epochs (Holdout Validation)...")
    model = LSTMAutoencoder().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(EPOCHS):
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
        print(f"Epoch {epoch+1}/{EPOCHS} - Val Loss: {avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            
    print(f"Optimal number of epochs determined: {best_epoch} (Loss: {best_val_loss:.6f})")
    
    print(f"Phase 2: Retraining on full dataset for {best_epoch} epochs...")
    full_dataset = TensorDataset(torch.tensor(X_data, dtype=torch.float32))
    full_train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    final_model = LSTMAutoencoder().to(DEVICE)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=LEARNING_RATE)
    
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
        print(f"Full Train Epoch {epoch+1}/{best_epoch}, Loss: {total_loss/len(full_train_loader):.6f}")

    print("Extracting price vectors...")
    final_model.eval()
    
    extraction_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE * 2, shuffle=False)
    
    all_embeddings = []
    with torch.no_grad():
        for (batch_x,) in extraction_loader:
            batch_x = batch_x.to(DEVICE)
            _, emb = final_model(batch_x)
            all_embeddings.append(emb.cpu().numpy())
            
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    
    df_vecs = pd.DataFrame(all_embeddings)
    df_vecs['gemrate_id'] = ids_map
    
    mean_vecs = df_vecs.groupby('gemrate_id').mean()
    
    return mean_vecs

def build_search_matrices(price_vecs, embedding_file):
    print("Loading Text Embeddings...")
    text_emb_df = pd.read_parquet(embedding_file)
    
    print(f"Stacking text embeddings for {len(text_emb_df)} items...")
    
    if 'gemrate_id' in text_emb_df.columns:
        text_emb_df = text_emb_df.set_index('gemrate_id')
        
    matrix = np.array(text_emb_df['embedding_vector'].tolist())
    
    norm = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    matrix_norm = matrix / norm
    
    text_embeddings_norm = pd.DataFrame(
        matrix_norm, 
        index=text_emb_df.index
    )

    print("Preparing Price Embeddings...")
    p_matrix = price_vecs.values
    p_norm = np.linalg.norm(p_matrix, axis=1, keepdims=True) + 1e-10
    p_matrix_norm = p_matrix / p_norm
    
    price_embeddings_norm = pd.DataFrame(
        p_matrix_norm,
        index=price_vecs.index
    )
    
    common_ids = text_embeddings_norm.index.intersection(price_embeddings_norm.index)
    common_ids = common_ids.sort_values()
    
    print(f"Building Search Database with {len(common_ids)} items (intersection of Text & Price)...")
    
    db_text_np = text_embeddings_norm.loc[common_ids].values
    db_price_np = price_embeddings_norm.loc[common_ids].values
    
    db_text_tensor = torch.tensor(db_text_np, dtype=torch.float32, device=DEVICE)
    db_price_tensor = torch.tensor(db_price_np, dtype=torch.float32, device=DEVICE)

    print("Searcher ready.")
    return db_text_tensor, db_price_tensor, common_ids, text_embeddings_norm, price_embeddings_norm

def main():
    print("Loading sales data...")
    if not os.path.exists(SALES_DATA_FILE):
        print(f"File {SALES_DATA_FILE} not found.")
        return

    df_sales = pd.read_csv(SALES_DATA_FILE, usecols=['gemrate_id', 'grade', 'date', 'price'], dtype={'gemrate_id': str})
    
    df_norm = normalize_grades(df_sales)
    
    _, X_windows, ids_map = prepare_price_matrix(df_norm)
    
    if len(X_windows) == 0:
        print("No windows generated. Exiting.")
        return

    price_vecs = train_and_extract(X_windows, ids_map)
    print("Price Vectors Head:")
    print(price_vecs.head())
    
    if os.path.exists(EMBEDDINGS_FILE):
        db_text_tensor, db_price_tensor, db_ids, all_text_df, all_price_df = build_search_matrices(price_vecs, EMBEDDINGS_FILE)
    else:
        print(f"Embeddings file {EMBEDDINGS_FILE} not found.")
        return

    ids_to_process = all_text_df.index
    total_items = len(ids_to_process)
    print(f"Processing all {total_items} items...")

    q_text_np = all_text_df.values
    q_text = torch.tensor(q_text_np, dtype=torch.float32, device=DEVICE)
    
    q_price_np = all_price_df.reindex(ids_to_process, fill_value=0).values
    q_price = torch.tensor(q_price_np, dtype=torch.float32, device=DEVICE)
    
    print("Computing similarity matrix...")
    with torch.no_grad():
        sim_text = torch.matmul(q_text, db_text_tensor.T)
        sim_price = torch.matmul(q_price, db_price_tensor.T)
        
        total_sim = sim_text + sim_price
        
        k_val = min(N_NEIGHBORS + 1, total_sim.shape[1])
        print(f"Finding top {k_val} matches...")
        top_vals, top_idxs = torch.topk(total_sim, k=k_val, dim=1)
        
        top_idxs = top_idxs.cpu().numpy()
        top_vals = top_vals.cpu().numpy()
    
    print("Processing results...")
    all_gemrate_ids = []
    all_neighbors = []
    all_scores = []
    
    for local_idx, query_id in enumerate(tqdm(ids_to_process, desc="Formatting output")):
        indices = top_idxs[local_idx]
        vals = top_vals[local_idx]
        
        neighbor_ids = db_ids[indices]
        
        mask = neighbor_ids != query_id
        final_ids = neighbor_ids[mask][:N_NEIGHBORS]
        final_scores = vals[mask][:N_NEIGHBORS]
        
        count = len(final_ids)
        all_gemrate_ids.extend([query_id] * count)
        all_neighbors.extend(final_ids)
        all_scores.extend(final_scores)
        
    results_df = pd.DataFrame({
        'gemrate_id': all_gemrate_ids,
        'neighbors': all_neighbors,
        'score': all_scores
    })

    print(f"Saving final output with {len(results_df)} rows...")
    results_df.to_parquet(OUTPUT_FILE)
    print("Done.")

if __name__ == "__main__":
    main()
