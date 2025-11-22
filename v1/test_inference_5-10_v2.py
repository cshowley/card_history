import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd


class FullGradeHistoryPredictor(nn.Module):
    """Model that uses ALL grade histories to predict a single target grade"""
    
    def __init__(self, n_cards, card_embed_dim=64, time_hidden=128, n_grades=6):
        """
        Args:
            n_cards: Total number of unique cards
            card_embed_dim: Dimension of card identity embeddings
            time_hidden: Hidden size for temporal processing
            n_grades: Number of possible grades (0-10 = 6)
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
        
        # Pack and process
        packed = nn.utils.rnn.pack_padded_sequence(
            gru_input, 
            reshaped_lengths.cpu(), 
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


def predict_price(model, card_id, target_grade, sales_records, card_to_idx, max_seq_len=5, device='cpu'):
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


def format_full_grade_inference(
    card_id,
    target_grade,
    sales_records,
    card_to_idx,
    max_seq_len,
    n_grades=6
):
    """
    Converts raw sales records to full grade history format
    
    Args:
        card_id: Card SPEC_ID
        target_grade: Grade to predict
        sales_records: List of all sales for this card (any grade)
        card_to_idx: Card ID mapping
        max_seq_len: Maximum history length
        n_grades: Number of possible grades (0-10 = 6)
    
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


## load model

checkpoint = torch.load('card_price_model_5-10.pth', weights_only=False)
# Extract the correct time_hidden value from the GRU weights
# For a GRU: weight_hh_l0 shape is [3*hidden_size, hidden_size]
# So hidden_size = weight_hh_l0.shape[1]
gru_weight_hh = checkpoint['model_state_dict']['gru.weight_hh_l0']
hidden_size = gru_weight_hh.shape[1]  # This will be 128
# Extract card embedding dimension
embed_dim = checkpoint['model_state_dict']['card_embed.weight'].shape[1]
# Now create a model with the EXACT same architecture
model = FullGradeHistoryPredictor(
    n_cards=len(checkpoint['card_to_idx']) + 2,  # Must match training
    card_embed_dim=embed_dim,                    # Extracted from checkpoint
    time_hidden=hidden_size,                     # Extracted from checkpoint (128)
    n_grades=6
)

# Now load the weights (this will work)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
# Optional: Extract other useful metadata
card_to_idx = checkpoint['card_to_idx']

raw_df = pd.read_csv('full_training_data_grades_5-10.csv')
card_ids = [3700060, 544027, 9656727, 7869313, 9603030]
card_id = 544028
grade = [i for i in range(0,6)]

tmp = raw_df[raw_df['spec_id'] == card_id]
tmp['date'] = pd.to_datetime(tmp['date'])
tmp = tmp.sort_values(by='date')
for target_grade in grade:

    card_sales = tmp.to_dict('records')
    sales_records = [{
        'grade': r['gradeNumber'],
        'date': pd.to_datetime(r['date']),
        'price': r['price']
    } for r in card_sales]

    # Predict price
    prediction = predict_price(
        model, 
        card_id, 
        target_grade, 
        sales_records, 
        card_to_idx,
        max_seq_len=8,
        device='cpu'
    )
    print(f'Grade {target_grade+5} prediction: {prediction}')
print(card_id)