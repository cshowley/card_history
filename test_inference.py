## load model

checkpoint = torch.load('card_price_model.pth', weights_only=False)
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
    n_grades=11
)

# Now load the weights (this will work)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
# Optional: Extract other useful metadata
card_to_idx = checkpoint['card_to_idx']

raw_df = pd.read_csv('training_data.csv')
card_id = 
target_grade = 

card_sales = raw_df[raw_df['spec_id'] == card_id].to_dict('records')
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