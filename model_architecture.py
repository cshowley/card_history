import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
        # Grade relationship encoder (critical for full history)
        self.grade_encoder = nn.Sequential(
            nn.Linear(time_hidden * n_grades, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(256, 128)
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
        
        # 3. Encode grade relationships
        # Flatten grade embeddings for each card
        flattened_grade_embs = grade_embs.view(batch_size, -1)
        grade_relationship = self.grade_encoder(flattened_grade_embs)  # [batch, 128]
        
        # 4. Combine with card identity for final prediction
        x = torch.cat([grade_relationship, card_emb], dim=1)
        return self.price_head(x).squeeze()