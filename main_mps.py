#!/usr/bin/env python3
"""
Trading Card Price Prediction System
$===================================$

This script implements a full pipeline for predicting graded trading card prices
using historical sales data across multiple grades. The system:
- Processes raw card sales database records
- Trains a model that leverages full grade histories
- Provides accurate price predictions while enforcing grade hierarchy

Usage:
    python main.py [--max-seq-len N] [--epochs N] [--batch-size N]

Example:
    python main.py --max-seq-len 8 --epochs 50 --batch-size 64
"""

import os
import sys
import argparse
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import copy
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("card_price_prediction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Select for cuda/mps/cpu
def get_optimal_device():
    """Determine the best available device (MPS > CUDA > CPU)"""
    # Check for MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        logger.info("Apple Silicon GPU (MPS) is available and will be used")
        return 'mps'
    # Check for CUDA (NVIDIA)
    elif torch.cuda.is_available():
        logger.info("CUDA GPU is available and will be used")
        return 'cuda'
    # Fall back to CPU
    else:
        logger.info("No GPU acceleration available, using CPU")
        return 'cpu'

# ======================$
# MODEL ARCHITECTURE
# ======================$

class FullGradeHistoryPredictor(nn.Module):
    """Model that uses ALL grade histories to predict a single target grade"""
    
    def __init__(self, n_cards, card_embed_dim=64, time_hidden=128, n_grades=11):
        """
        Args:
            n_cards: Total number of unique cards
            card_embed_dim: Dimension of card identity embeddings
            time_hidden: Hidden size for temporal processing
            n_grades: Number of possible grades (0-10 = 11)
        """
        super().__init__()
        self.n_grades = n_grades
        self.card_embed = nn.Embedding(n_cards, card_embed_dim, padding_idx=0)
        
        # Shared temporal processor for ALL grades
        self.gru = nn.GRU(
            input_size=2,  # [price, decay_weight]
            hidden_size=time_hidden,
            batch_first=True,
            num_layers=1,
            dropout=0.1
        )
        
        # Final prediction head
        self.price_head = nn.Sequential(
            nn.Linear(128 + card_embed_dim, 192),
            nn.LayerNorm(192),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(192, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1)
        )
    
    def forward(self, card_ids, target_grades, all_prices, all_gaps, all_lengths):
        """
        Args:
            card_ids: [batch] - Card SPEC_IDs (mapped to indices)
            target_grades: [batch] - Target grades to predict (0-10)
            all_prices: [batch, n_grades, seq_len] - Prices for ALL grades
            all_gaps: [batch, n_grades, seq_len] - Days since last sale for ALL grades
            all_lengths: [batch, n_grades] - Actual sequence lengths for ALL grades
            
        Returns:
            Predicted prices [batch]
        """
        batch_size = card_ids.size(0)
        
        # 1. Card identity embedding
        card_emb = self.card_embed(card_ids)  # [batch, card_dim]
        
        # 2. Process ALL grade histories through shared GRU
        # Reshape to [batch*n_grades, seq_len, 2]
        batch_n_grades = batch_size * self.n_grades
        reshaped_prices = all_prices.view(batch_n_grades, -1)
        reshaped_gaps = all_gaps.view(batch_n_grades, -1)
        reshaped_lengths = all_lengths.view(-1)

        # Apply decay weights
        decay_weights = torch.exp(-0.05 * reshaped_gaps)
        gru_input = torch.stack([reshaped_prices, decay_weights], dim=-1)
        
        # DETERMINE DEVICE FROM INPUT TENSORS (CORRECT APPROACH)
        device = card_ids.device
        
        # Handle MPS-specific requirements
        if device.type == 'mps':
            # MPS requires lengths to be on CPU
            packed = nn.utils.rnn.pack_padded_sequence(
                gru_input, 
                reshaped_lengths.cpu(),
                batch_first=True, 
                enforce_sorted=False
            )
        else:
            packed = nn.utils.rnn.pack_padded_sequence(
                gru_input, 
                reshaped_lengths.to(device),
                batch_first=True, 
                enforce_sorted=False
            )
        
        _, h_n = self.gru(packed)
        grade_embs = h_n.squeeze(0)  # [batch*n_grades, time_hidden]
        
        # Reshape back to [batch, n_grades, time_hidden]
        grade_embs = grade_embs.view(batch_size, self.n_grades, -1)
        
        # 3. Extract target grade representations
        # Convert target_grades to long tensor for indexing
        target_indices = target_grades.long().unsqueeze(-1).unsqueeze(-1).expand(-1, 1, grade_embs.size(2))
        # Gather the specific grade embedding for each sample
        target_grade_embs = grade_embs.gather(1, target_indices).squeeze(1)  # [batch, time_hidden]

        # 4. Combine with card identity for final prediction
        x = torch.cat([target_grade_embs, card_emb], dim=1)
        return self.price_head(x).squeeze()

# ======================$
# DATA PREPROCESSING
# ======================$

def preprocess_for_full_grade(raw_df, max_seq_len=12, test_size=0.15):
    """
    Creates training data that includes ALL grade histories
    
    Critical: For each sale, we inject ALL available grade histories
    """
    logger.info("Starting data preprocessing...")
    df = raw_df.copy()
    
    # Rename columns for clarity
    df = df.rename(columns={
        'spec_id': 'card_id',
        'date': 'sale_date',
        'price': 'price',
        'gradeNumber': 'grade'
    })
    
    # Filter and clean
    df = df[df['price'] > 0]
    df['sale_date'] = pd.to_datetime(df['sale_date'])
    df = df.sort_values(['card_id', 'grade', 'sale_date'])
    
    # Create card ID mapping
    unique_cards = sorted(df['card_id'].unique())
    card_to_idx = {card: i+1 for i, card in enumerate(unique_cards)}
    idx_to_card = {i: card for card, i in card_to_idx.items()}
    
    logger.info(f"Found {len(unique_cards)} unique cards in dataset")
    
    # Calculate time gaps per card-grade
    df['prev_date'] = df.groupby(['card_id', 'grade'])['sale_date'].shift(1)
    df['days_gap'] = (df['sale_date'] - df['prev_date']).dt.days
    df['days_gap'] = df['days_gap'].fillna(0).clip(lower=0)
    
    # Build training sequences
    all_data = []
    
    # Process each card
    for card_id, card_group in tqdm(df.groupby('card_id'), desc="Processing cards"):
        card_idx = card_to_idx[card_id]
        
        # For each grade, collect all sales
        grade_histories = {}
        for grade in range(0, 11):
            grade_sales = card_group[card_group['grade'] == grade]
            if not grade_sales.empty:
                grade_histories[grade] = {
                    'prices': grade_sales['price'].values,
                    'gaps': grade_sales['days_gap'].values,
                    'dates': grade_sales['sale_date'].values
                }
        
        # For each sale across all grades
        for grade, history in grade_histories.items():
            prices = history['prices']
            gaps = history['gaps']
            dates = history['dates']
            
            # Create N-1 training examples per grade
            for i in range(1, len(prices)):
                # Build grade histories as of this sale date
                current_date = dates[i]
                
                # Initialize lists to collect data for grades with sales
                valid_grades = []
                grade_prices = []
                grade_gaps = []
                grade_lengths = []
                
                # For each possible grade
                for g in range(0, 11):
                    if g in grade_histories:
                        g_sales = grade_histories[g]
                        mask = g_sales['dates'] <= current_date
                        g_prices = g_sales['prices'][mask]
                        g_gaps = g_sales['gaps'][mask]
                        
                        n_sales = len(g_prices)
                        if n_sales > 0:
                            valid_grades.append(g)
                            grade_lengths.append(min(n_sales, max_seq_len))
                            grade_prices.append(g_prices[-min(n_sales, max_seq_len):])
                            grade_gaps.append(g_gaps[-min(n_sales, max_seq_len):])
                
                # Skip if no grades have sales history (shouldn't happen but safety check)
                if len(valid_grades) == 0:
                    continue
                
                # Create padded arrays only for valid grades
                n_valid_grades = len(valid_grades)
                all_prices = np.zeros((11, max_seq_len))  # Still 11 grades but...
                all_gaps = np.zeros((11, max_seq_len))
                all_lengths = np.zeros(11, dtype=int)
                
                # Initialize all lengths to 1 (minimum required by PyTorch)
                all_lengths[:] = 1
                
                # Fill in data only for valid grades
                for idx, (g, prices_list, gaps_list, length) in enumerate(zip(
                    valid_grades, grade_prices, grade_gaps, grade_lengths)):
                    all_lengths[g] = length
                    all_prices[g, :length] = prices_list
                    all_gaps[g, :length] = gaps_list
                
                # Store training example
                all_data.append({
                    'card_idx': card_idx,
                    'target_grade': float(grade),
                    'all_prices': all_prices,
                    'all_gaps': all_gaps,
                    'all_lengths': all_lengths,
                    'target_price': float(prices[i]),
                    'card_id': card_id,
                    'grade': float(grade),
                    'date': dates[i]
                })    

    if not all_data:
        raise ValueError("No valid training examples generated. Check your data format and filters.")
    
    logger.info(f"Generated {len(all_data)} training examples")
    
    # Split data
    full_df = pd.DataFrame(all_data)
    train_df, val_df = train_test_split(
        full_df, 
        test_size=test_size, 
        random_state=42,
    )
    
    # Create Dataset objects
    train_dataset = FullGradeDataset(train_df, max_seq_len)
    val_dataset = FullGradeDataset(val_df, max_seq_len)
    
    return train_dataset, val_dataset, card_to_idx, idx_to_card

class FullGradeDataset(Dataset):
    """Dataset that includes full grade histories"""
    
    def __init__(self, df, max_seq_len):
        self.df = df
        self.max_seq_len = max_seq_len
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        return {
            'card_ids': torch.tensor(row['card_idx'], dtype=torch.long),
            'target_grades': torch.tensor(row['target_grade'], dtype=torch.float32),
            'all_prices': torch.tensor(row['all_prices'], dtype=torch.float32),
            'all_gaps': torch.tensor(row['all_gaps'], dtype=torch.float32),
            'all_lengths': torch.tensor(row['all_lengths'], dtype=torch.long),
            'targets': torch.tensor(row['target_price'], dtype=torch.float32)
        }

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    
    def __init__(self, patience=15, min_delta=0.0001, verbose=True):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 15 (matches current code)
            min_delta (float): Minimum change in validation loss to qualify as improvement
            verbose (bool): If True, logs messages about stopping criteria
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model, save_path, metadata):
        """
        Call method to check if early stopping should be triggered.
        
        Args:
            val_loss (float): Current validation loss
            model (nn.Module): The model being trained
            save_path (str): Path to save the best model
            metadata (dict): Additional metadata to save with the model
            
        Returns:
            bool: True if training should stop, False otherwise
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, save_path, metadata)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, save_path, metadata)
            self.counter = 0
            
        return self.early_stop

    def save_checkpoint(self, val_loss, model, save_path, metadata):
        '''Saves model with all necessary metadata when validation loss decreases.'''
        if self.verbose:
            logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        
        # Save complete model state with metadata
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss
        }
        checkpoint.update(metadata)  # Add all metadata fields
        
        torch.save(checkpoint, save_path)
        self.val_loss_min = val_loss

def full_grade_collate(batch):
    """Custom collate function for full grade histories"""
    card_ids = torch.stack([item['card_ids'] for item in batch])
    target_grades = torch.stack([item['target_grades'] for item in batch])
    all_prices = torch.stack([item['all_prices'] for item in batch])
    all_gaps = torch.stack([item['all_gaps'] for item in batch])
    all_lengths = torch.stack([item['all_lengths'] for item in batch])
    targets = torch.stack([item['targets'] for item in batch])
    
    return {
        'card_ids': card_ids,
        'target_grades': target_grades,
        'all_prices': all_prices,
        'all_gaps': all_gaps,
        'all_lengths': all_lengths,
        'targets': targets
    }

# ======================$
# TRAINING & EVALUATION
# ======================$

def calculate_mape(predictions, targets):
    """Calculate Mean Absolute Percentage Error"""
    return 100 * torch.mean(torch.abs((targets - predictions) / targets))

def calculate_full_grade_loss(predictions, targets, all_lengths, all_prices, model,
                            mse_weight=1.0, monotonicity_weight=1.5):
    """Improved loss with grade monotonicity constraint"""
    # 1. Primary MSE loss
    mse_loss = F.mse_loss(predictions, targets)
    
    # 2. Grade monotonicity constraint
    # For each card, ensure higher grades have higher predicted prices
    monotonicity_loss = 0.0
    batch_size = all_prices.size(0)
    
    for i in range(batch_size):
        # Get the sequence lengths for this card
        lengths = all_lengths[i]
        
        # For each grade that has at least 2 sales
        for grade in range(1, 11):  # Start from grade 1 since we compare with grade-1
            if lengths[grade] > 0 and lengths[grade-1] > 0:
                # Get the most recent price for each grade
                grade_price = all_prices[i, grade, lengths[grade]-1]
                prev_grade_price = all_prices[i, grade-1, lengths[grade-1]-1]
                
                # If higher grade has lower price, add penalty
                if grade_price < prev_grade_price:
                    monotonicity_loss += (prev_grade_price - grade_price)
    
    # Normalize the monotonicity loss
    if batch_size > 0:
        monotonicity_loss /= batch_size
    
    return mse_weight * mse_loss + monotonicity_weight * monotonicity_loss

def train_full_grade_model(
    model,
    train_loader,
    val_loader,
    epochs=200,
    lr=0.001,
    device='auto',
    save_path='card_price_model.pth'
):
    # Resolve device if 'auto' is specified
    if device == 'auto':
        device = get_optimal_device()
    
    # Move MPS initialization HERE (before moving model to device)
    if device == 'mps':
        torch.mps.empty_cache()
        torch.mps.set_per_process_memory_fraction(0.8)
    
    logger.info(f"Starting training on {device}...")
    model = model.to(device)

    """Train the Full Grade History model"""
    logger.info(f"Starting training on {device}...")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5
    )
    
    # Prepare metadata template for saving
    metadata = {
        'card_to_idx': train_loader.dataset.card_to_idx,
        'idx_to_card': train_loader.dataset.idx_to_card,
        'max_seq_len': train_loader.dataset.max_seq_len,
    }
    
    # Initialize EarlyStopping callback
    early_stopping = EarlyStopping(patience=15, min_delta=0.0001, verbose=True)

    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_mape = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            outputs = model(
                card_ids=batch['card_ids'],
                target_grades=batch['target_grades'],
                all_prices=batch['all_prices'],
                all_gaps=batch['all_gaps'],
                all_lengths=batch['all_lengths']
            )
            
            loss = calculate_full_grade_loss(
                outputs, 
                batch['targets'],
                batch['all_lengths'],
                batch['all_prices'],
                model
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_mape += calculate_mape(outputs, batch['targets']).item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mape = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(
                    card_ids=batch['card_ids'],
                    target_grades=batch['target_grades'],
                    all_prices=batch['all_prices'],
                    all_gaps=batch['all_gaps'],
                    all_lengths=batch['all_lengths']
                )
                
                val_loss += F.mse_loss(outputs, batch['targets']).item()
                val_mape += calculate_mape(outputs, batch['targets']).item()
        
        # Average metrics
        train_loss /= len(train_loader)
        train_mape /= len(train_loader)
        val_loss /= len(val_loader)
        val_mape /= len(val_loader)
        
        scheduler.step(val_loss)
        
        logger.info(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Train MAPE: {train_mape:.2f}% | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val MAPE: {val_mape:.2f}%")
        
        # Update metadata with current metrics
        current_metadata = metadata.copy()
        current_metadata.update({
            'val_mape': val_mape,
            'train_mape': train_mape,
            'epoch': epoch + 1
        })
        
        # NEW: Check early stopping using callback pattern
        if early_stopping(val_loss, model, save_path, current_metadata):
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load the best model
    if device == 'mps':
        checkpoint = torch.load(save_path, map_location='mps', weights_only=False)
    else:
        checkpoint = torch.load(save_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model.state_dict()

def evaluate_model(model, test_loader, device='auto'):
    if device == 'auto':
        device = get_optimal_device()
    
    """Evaluate model on test set"""
    logger.info(f"Evaluating model on {device}...")
    model = model.to(device)

    model.eval()
    
    total_mape = 0.0
    total_samples = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluation"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                card_ids=batch['card_ids'],
                target_grades=batch['target_grades'],
                all_prices=batch['all_prices'],
                all_gaps=batch['all_gaps'],
                all_lengths=batch['all_lengths']
            )
            
            batch_mape = calculate_mape(outputs, batch['targets'])
            total_mape += batch_mape.item() * len(batch['targets'])
            total_samples += len(batch['targets'])
            
            predictions.extend(outputs.cpu().numpy())
            targets.extend(batch['targets'].cpu().numpy())
    
    avg_mape = total_mape / total_samples if total_samples > 0 else float('nan')
    
    # Calculate error by grade
    grade_errors = {}
    for grade in range(0, 11):
        grade_mask = np.array([t['target_grade'] == grade for _, t in enumerate(test_loader.dataset.df.iloc)])
        if np.any(grade_mask):
            grade_preds = np.array(predictions)[grade_mask]
            grade_tgts = np.array(targets)[grade_mask]
            grade_errors[grade] = 100 * np.mean(np.abs((grade_tgts - grade_preds) / grade_tgts))
    
    logger.info(f"Test MAPE: {avg_mape:.2f}%")
    for grade, error in sorted(grade_errors.items()):
        logger.info(f"  Grade {grade}: {error:.2f}% MAPE")
    
    return avg_mape, grade_errors

# ======================$
# INFERENCE UTILITIES
# ======================$

def format_full_grade_inference(
    card_id,
    target_grade,
    sales_records,
    card_to_idx,
    max_seq_len,
    n_grades=11
):
    """
    Converts raw sales records to full grade history format
    
    Args:
        card_id: Card SPEC_ID
        target_grade: Grade to predict
        sales_records: List of all sales for this card (any grade)
        card_to_idx: Card ID mapping
        max_seq_len: Maximum history length
        n_grades: Number of possible grades (0-10 = 11)
    
    Returns:
        Dictionary of tensors for model inference
    """
    # Organize records by grade
    grade_histories = {g: [] for g in range(n_grades)}
    
    for record in sales_records:
        grade = int(record['grade'])
        if 0 <= grade < n_grades:
            grade_histories[grade].append({
                'date': record['date'],
                'price': record['price']
            })
    
    # Sort each grade history chronologically
    for grade in grade_histories:
        grade_histories[grade] = sorted(
            grade_histories[grade], 
            key=lambda x: x['date']
        )
    
    # Build all_prices, all_gaps, all_lengths
    all_prices = np.zeros((n_grades, max_seq_len))
    all_gaps = np.zeros((n_grades, max_seq_len))
    all_lengths = np.zeros(n_grades, dtype=int)
    
    for grade in range(n_grades):
        history = grade_histories[grade]
        n_sales = len(history)
        
        if n_sales > 0:
            # Store prices
            prices = [sale['price'] for sale in history]
            all_prices[grade, :n_sales] = prices[-max_seq_len:]
            
            # Calculate gaps
            gaps = [0.0]
            for i in range(1, n_sales):
                gap = (history[i]['date'] - history[i-1]['date']).days
                gaps.append(float(gap))
            
            all_gaps[grade, :n_sales] = gaps[-max_seq_len:]
            all_lengths[grade] = min(n_sales, max_seq_len)
    
    # Get card index with special handling for new cards
    if card_id in card_to_idx:
        card_idx = card_to_idx[card_id]
    else:
        # Create a special "unknown card" index (last index)
        card_idx = len(card_to_idx)  # This will be n_cards (not n_cards+1 since 0 is padding)
        
        # If this is the first unknown card we've seen, log a warning
        if not hasattr(format_full_grade_inference, 'unknown_card_warning'):
            logger.warning("Encountered card ID not in training data. Using special 'unknown card' embedding.")
            format_full_grade_inference.unknown_card_warning = True
    
    return {
        "card_ids": torch.tensor([card_idx], dtype=torch.long),
        "target_grades": torch.tensor([float(target_grade)], dtype=torch.float32),
        "all_prices": torch.tensor([all_prices], dtype=torch.float32),
        "all_gaps": torch.tensor([all_gaps], dtype=torch.float32),
        "all_lengths": torch.tensor([all_lengths], dtype=torch.long)
    }

def predict_price(model, card_id, target_grade, sales_records, card_to_idx, max_seq_len=5, device='auto'):
    if device == 'auto':
        device = get_optimal_device()
    
    # Load model with appropriate device mapping
    if device == 'mps' and next(model.parameters()).device.type == 'cpu':
        model = model.to('mps')
    elif device == 'cpu' and next(model.parameters()).device.type == 'mps':
        model = model.to('cpu')
    """Predict price for a card with given sales history"""
    model.eval()
    model = model.to(device)

    # Format input
    inference_input = format_full_grade_inference(
        card_id=card_id,
        target_grade=target_grade,
        sales_records=sales_records,
        card_to_idx=card_to_idx,
        max_seq_len=max_seq_len
    )

    # Move to device
    inference_input = {k: v.to(device) for k, v in inference_input.items()}

    # Check for insufficient data
    all_gaps = inference_input['all_gaps']
    if torch.sum(all_gaps) == 0:
        return "Insufficient data: Card has 1 or 0 total sales over its history"

    # Run inference
    with torch.no_grad():
        prediction = model(**inference_input)

    return prediction.item()

# ======================$
# SYNTHETIC DATA GENERATION (FOR DEMO)
# ======================$

def generate_synthetic_card_data(n_cards=50, n_sales_per_card=30, seed=42):
    """Generate synthetic card sales data for demonstration"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    logger.info(f"Generating synthetic data for {n_cards} cards...")
    
    records = []
    current_date = datetime.datetime(2021, 1, 1)
    end_date = datetime.datetime(2023, 12, 31)
    
    # Base prices for different card types
    card_type_bases = {
        0: 10.0,    # Common modern
        1: 50.0,    # Uncommon modern
        2: 200.0,   # Rare modern
        3: 1000.0,  # Vintage common
        4: 5000.0   # Vintage rare
    }
    
    # Grade premiums (higher grades have higher premiums)
    grade_premiums = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.5, 2.5, 4.0]
    
    for card_id in range(1, n_cards + 1):
        # Randomly assign card type
        card_type = np.random.choice(list(card_type_bases.keys()))
        base_price = card_type_bases[card_type]
        
        # Determine if card is vintage (more volatility)
        is_vintage = card_type >= 3
        
        # Generate sales history
        sale_date = current_date
        last_prices = {g: base_price * (1 + grade_premiums[g]) for g in range(11)}
        
        for _ in range(n_sales_per_card):
            # Random grade (weighted toward middle grades)
            grade_probs = [0.01, 0.02, 0.05, 0.08, 0.12, 0.15, 0.18, 0.15, 0.12, 0.08, 0.04]
            grade = np.random.choice(range(11), p=grade_probs)
            
            # Price movement (more volatile for vintage cards)
            volatility = 0.15 if is_vintage else 0.08
            price_change = np.random.normal(0, volatility)
            
            # Apply price change to all grades (higher grades more affected)
            for g in range(11):
                last_prices[g] *= (1 + price_change * (g / 10.0))
            
            # Add some random noise to this grade's price
            noise = np.random.normal(0, 0.03)
            price = last_prices[grade] * (1 + noise)
            
            # Ensure minimum price
            price = max(price, 1.0)
            
            # Add record
            records.append({
                "_id": f"synth_{card_id}_{len(records)}",
                "spec_id": card_id,
                "date": sale_date,
                "price": round(price, 2),
                "gradeNumber": float(grade),
                "gradingCompany": "PSA"
            })
            
            # Move to next sale date (random interval)
            days_until_next = np.random.randint(3, 60) if not is_vintage else np.random.randint(7, 120)
            sale_date += datetime.timedelta(days=days_until_next)
            
            # Stop if we've gone beyond end date
            if sale_date > end_date:
                break
    
    logger.info(f"Generated {len(records)} synthetic sales records")
    return pd.DataFrame(records)

# ======================$
# MAIN EXECUTION
# ======================$

def main(args):
    """Main function that orchestrates the entire pipeline"""
    # 1. Generate or load data
    if args.synthetic:
        raw_df = generate_synthetic_card_data(
            n_cards=args.n_cards,
            n_sales_per_card=args.sales_per_card
        )
    else:
        if not os.path.exists(args.data_path):
            raise FileNotFoundError(f"Data file not found: {args.data_path}")
        
        logger.info(f"Loading data from {args.data_path}")
        raw_df = pd.read_csv(args.data_path)
    
    # 2. Preprocess data
    train_dataset, val_dataset, card_to_idx, idx_to_card = preprocess_for_full_grade(
        raw_df,
        max_seq_len=args.max_seq_len
    )
    
    # Add metadata to datasets for later use
    train_dataset.card_to_idx = card_to_idx
    train_dataset.idx_to_card = idx_to_card
    train_dataset.max_seq_len = args.max_seq_len
    
    val_dataset.card_to_idx = card_to_idx
    val_dataset.idx_to_card = idx_to_card
    val_dataset.max_seq_len = args.max_seq_len
    
    # 3. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=full_grade_collate,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=full_grade_collate,
        num_workers=0
    )
    
    logger.info(f"Created DataLoaders: train={len(train_loader)}, val={len(val_loader)}")
    
    # 4. Initialize model
    model = FullGradeHistoryPredictor(
        n_cards=len(card_to_idx) + 2,  # +1 for padding index +1 for unknown cards
        card_embed_dim=args.embed_dim,
        time_hidden=args.time_hidden,
        n_grades=11
    )
    
    logger.info(f"Initialized model with {len(card_to_idx)} unique cards")
    
    # Resolve device
    if args.device == 'auto':
        device = get_optimal_device()
    else:
        device = args.device
        logger.info(f"Using explicitly specified device: {device}")

    # 5. Train model
    best_model_state = train_full_grade_model(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        save_path=args.model_path
    )

    # 6. Load best model for evaluation
    model.load_state_dict(best_model_state)
    
    # 7. Evaluate model
    test_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=full_grade_collate,
        num_workers=0
    )
    
    avg_mape, grade_errors = evaluate_model(model, test_loader, device=args.device)
    
    # # 8. Demonstrate inference with different scenarios
    # logger.info("\n" + "="*50)
    # logger.info("INFERENCE DEMONSTRATIONS")
    # logger.info("="*50)
    

    # # Helper function to sample card_id from training data
    # ### FIX: model crashes if new card w/ no sales data is served for inference
    # def sample_spec_id():
    #     sampled_id = raw_df.spec_id.sample(1).astype(float).iloc[0]        
    #     return sampled_id


    # # Example 1: Standard case with multiple grade histories
    # logger.info("\nExample 1: Standard case (multiple grade histories)")
    # card_id = sample_spec_id()  # First card in synthetic data
    # target_grade = 9
    
    # # Get all sales for this card from raw data
    # card_sales = raw_df[raw_df['spec_id'] == card_id].to_dict('records')
    # sales_records = [{
    #     'grade': r['gradeNumber'],
    #     'date': pd.to_datetime(r['date']),
    #     'price': r['price']
    # } for r in card_sales]

    # # Predict price
    # prediction = predict_price(
    #     model, 
    #     card_id, 
    #     target_grade, 
    #     sales_records, 
    #     card_to_idx,
    #     max_seq_len=args.max_seq_len,
    #     device=args.device
    # )
    
    # # Get actual recent price for comparison (if available)
    # recent_sales = [r for r in sales_records if r['grade'] == target_grade]
    # actual_price = recent_sales[-1]['price'] if recent_sales else "N/A"
    
    # logger.info(f"Predicted price for card {card_id}, grade {target_grade}: ${prediction:.2f}")
    # logger.info(f"Most recent actual price: {actual_price}")
    
    # # Example 2: Sparse data case
    # logger.info("\nExample 2: Sparse data case (infrequent sales)")
    # card_id = sample_spec_id()  # Another card
    
    # # Get sales but only keep the first and last
    # card_sales = raw_df[raw_df['spec_id'] == card_id].to_dict('records')
    # if len(card_sales) > 2:
    #     card_sales = [card_sales[0], card_sales[-1]]
    
    # sales_records = [{
    #     'grade': r['gradeNumber'],
    #     'date': pd.to_datetime(r['date']),
    #     'price': r['price']
    # } for r in card_sales]
    
    # # Predict price for grade 8
    # prediction = predict_price(
    #     model, 
    #     card_id, 
    #     8, 
    #     sales_records, 
    #     card_to_idx,
    #     max_seq_len=args.max_seq_len,
    #     device=args.device
    # )
    
    # try:
    #     logger.info(f"Predicted price for sparse card {card_id}, grade 8: ${prediction:.2f}")
    # except ValueError:
    #     logger.info(f"Insufficient data for prediction")        

    # # Example 3: New card with no sales history
    # logger.info("\nExample 3: New card with no sales history")
    # new_card_id = sample_spec_id()  # Not in training data
    
    # # Empty sales history
    # sales_records = []
    
    # # Predict price for grade 7
    # prediction = predict_price(
    #     model, 
    #     new_card_id, 
    #     7, 
    #     sales_records, 
    #     card_to_idx,
    #     max_seq_len=args.max_seq_len,
    #     device=args.device
    # )
    
    # try:
    #     logger.info(f"Predicted price for new card {new_card_id}, grade 7: ${prediction:.2f}")
    # except ValueError:
    #     logger.info(f"Insufficient data for prediction")        
    
    # # Example 4: Dormant high grade
    # logger.info("\nExample 4: Dormant high grade (grade 10 hasn't sold recently)")
    # card_id = sample_spec_id()  # Another card
    
    # # Get sales but remove recent grade 10 sales
    # card_sales = raw_df[raw_df['spec_id'] == card_id].to_dict('records')
    # filtered_sales = []
    # last_grade_10_date = None
    
    # # Find last grade 10 sale
    # for r in reversed(card_sales):
    #     if r['gradeNumber'] == 10.0:
    #         last_grade_10_date = pd.to_datetime(r['date'])
    #         break
    
    # # Keep only grade 10 sales older than 90 days
    # if last_grade_10_date:
    #     cutoff_date = last_grade_10_date - datetime.timedelta(days=90)
    #     for r in card_sales:
    #         grade = r['gradeNumber']
    #         date = pd.to_datetime(r['date'])
            
    #         if grade != 10.0 or date < cutoff_date:
    #             filtered_sales.append(r)
    # else:
    #     filtered_sales = card_sales
    
    # sales_records = [{
    #     'grade': r['gradeNumber'],
    #     'date': pd.to_datetime(r['date']),
    #     'price': r['price']
    # } for r in filtered_sales]
    
    # # Predict price for grade 10
    # prediction = predict_price(
    #     model, 
    #     card_id, 
    #     10, 
    #     sales_records, 
    #     card_to_idx,
    #     max_seq_len=args.max_seq_len,
    #     device=args.device
    # )
    
    # logger.info(f"Predicted price for card {card_id}, grade 10 (dormant): ${prediction:.2f}")
    
    # logger.info("\nTraining pipeline completed successfully!")
    # logger.info(f"Model saved to {args.model_path}")
    # logger.info(f"Final validation MAPE: {avg_mape:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trading Card Price Prediction System')
    
    # Data arguments
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data instead of real data')
    parser.add_argument('--data-path', type=str, default='card_sales.csv', 
                        help='Path to CSV file with card sales data')
    parser.add_argument('--n-cards', type=int, default=50, 
                        help='Number of cards to generate in synthetic data')
    parser.add_argument('--sales-per-card', type=int, default=30, 
                        help='Number of sales per card in synthetic data')
    
    # Model arguments
    parser.add_argument('--max-seq-len', type=int, default=8, 
                        help='Maximum sequence length for historical data')
    parser.add_argument('--embed-dim', type=int, default=64, 
                        help='Dimension of card embedding vectors')
    parser.add_argument('--time-hidden', type=int, default=128, 
                        help='Hidden size for temporal processing')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50, 
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, 
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use for training (mps/cuda/cpu). Use "auto" for automatic detection.')

    # Output arguments
    parser.add_argument('--model-path', type=str, default='card_price_model.pth',
                        help='Path to save the trained model')
    
    args = parser.parse_args()
    
    # Print configuration
    logger.info("Starting trading card price prediction pipeline with configuration:")
    for arg in vars(args):
        logger.info(f"  {arg}: {getattr(args, arg)}")
    
    try:
        main(args)
    except Exception as e:
        logger.exception("An error occurred during execution")
        sys.exit(1)