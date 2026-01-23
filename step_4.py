import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import constants
from data_integrity import get_tracker


class LSTMAutoencoder(nn.Module):
    def __init__(
        self,
        seq_len=constants.S4_WINDOW_SIZE,
        n_features=1,
        embedding_dim=constants.S4_PRICE_EMBEDDING_DIM,
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

    returns = pivot_df / pivot_df.shift(1)
    returns = returns.dropna()

    data_values = returns.values
    card_ids = returns.columns
    print("ids after drop", len(set(card_ids)))
    if data_values.shape[0] < constants.S4_WINDOW_SIZE:
        print("Not enough time points for windowing.")
        return pivot_df, [], []

    print("Generating sliding windows...")

    raw_windows = []
    for i in range(len(data_values) - constants.S4_WINDOW_SIZE + 1):
        window = data_values[i : i + constants.S4_WINDOW_SIZE, :]
        raw_windows.append(window)

    raw_windows = np.array(raw_windows)

    processed = np.transpose(raw_windows, (2, 0, 1))

    processed = processed.reshape(-1, constants.S4_WINDOW_SIZE)

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

    train_loader = DataLoader(
        train_ds, batch_size=constants.S4_BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(val_ds, batch_size=constants.S4_BATCH_SIZE, shuffle=False)

    print("Phase 1: Finding optimal epochs (Holdout Validation)...")
    model = LSTMAutoencoder().to(constants.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=constants.S4_LEARNING_RATE)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(constants.S4_EPOCHS):
        model.train()
        for (batch_x,) in train_loader:
            batch_x = batch_x.to(constants.DEVICE)
            optimizer.zero_grad()
            recon, _ = model(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for (batch_x,) in val_loader:
                batch_x = batch_x.to(constants.DEVICE)
                recon, _ = model(batch_x)
                loss = criterion(recon, batch_x)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}/{constants.S4_EPOCHS} - Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1

    print(
        f"Optimal number of epochs determined: {best_epoch} (Loss: {best_val_loss:.6f})"
    )

    print(f"Phase 2: Retraining on full dataset for {best_epoch} epochs...")
    full_dataset = TensorDataset(torch.tensor(X_data, dtype=torch.float32))
    full_train_loader = DataLoader(
        full_dataset, batch_size=constants.S4_BATCH_SIZE, shuffle=True
    )

    final_model = LSTMAutoencoder().to(constants.DEVICE)
    optimizer = torch.optim.Adam(
        final_model.parameters(), lr=constants.S4_LEARNING_RATE
    )

    final_model.train()
    for epoch in range(best_epoch):
        total_loss = 0
        for (batch_x,) in full_train_loader:
            batch_x = batch_x.to(constants.DEVICE)
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
        full_dataset, batch_size=constants.S4_BATCH_SIZE * 2, shuffle=False
    )

    all_embeddings = []
    with torch.no_grad():
        for (batch_x,) in extraction_loader:
            batch_x = batch_x.to(constants.DEVICE)
            _, emb = final_model(batch_x)
            all_embeddings.append(emb.cpu().numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)

    df_vecs = pd.DataFrame(all_embeddings)
    df_vecs["gemrate_id"] = ids_map

    mean_vecs = df_vecs.groupby("gemrate_id").mean()

    return mean_vecs


def run_step_4():
    start_time = time.time()
    tracker = get_tracker()

    df_sales = pd.read_parquet(
        constants.S3_HISTORICAL_DATA_FILE,
        columns=["gemrate_id", "grade", "date", "price"],
    )
    df_sales = df_sales.dropna(subset="price")

    df_norm = s4_normalize_grades(df_sales)
    _, X_windows, ids_map = s4_prepare_price_matrix(df_norm)

    if len(X_windows) == 0:
        print("No windows generated. Exiting.")
        return

    price_vecs = s4_train_and_extract(X_windows, ids_map)
    print("Price Vectors Head:")
    print(price_vecs.head())

    print(f"Saving price vectors to {constants.S4_OUTPUT_PRICE_VECS_FILE}...")
    price_vecs.to_parquet(constants.S4_OUTPUT_PRICE_VECS_FILE)

    # Data Integrity Tracking
    duration = time.time() - start_time
    tracker.add_metric(
        id="s4_cards_with_price_vectors",
        title="Cards with Price Vectors",
        value=f"{len(price_vecs):,}",
    )
    tracker.add_metric(
        id="s4_duration",
        title="Step 4 Duration",
        value=f"{duration:.1f}",
    )

    print("Step 4 Complete.")
