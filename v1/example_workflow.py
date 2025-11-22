# Card ID mapping
card_to_idx = {9656727: 42}

# Inference input for PSA 9 1986 Fleer MJ
inference_input = {
    "card_ids": torch.tensor([42]),       # Card ID mapped to index
    "target_grades": torch.tensor([9.0]), # Target grade to predict
    
    # ALL grade histories (0-10), but only grades 7-10 have data
    "all_prices": torch.tensor([[
        [0.0, 0.0, 0.0, 0.0, 0.0],  # Grade 0 (no data)
        [0.0, 0.0, 0.0, 0.0, 0.0],  # Grade 1
        # ... grades 2-6 (all zeros)
        [0.0, 0.0, 0.0, 0.0, 0.0],  # Grade 6
        [200.0, 220.0, 0.0, 0.0, 0.0],  # Grade 7 history
        [350.0, 375.0, 0.0, 0.0, 0.0],  # Grade 8 history
        [500.0, 520.0, 550.0, 0.0, 0.0],  # Grade 9 history
        [620.0, 0.0, 0.0, 0.0, 0.0]       # Grade 10 history
    ]]),
    
    "all_gaps": torch.tensor([[
        [0.0, 0.0, 0.0, 0.0, 0.0],  # Grade 0 gaps
        [0.0, 0.0, 0.0, 0.0, 0.0],  # Grade 1 gaps
        # ... grades 2-6 (all zeros)
        [0.0, 0.0, 0.0, 0.0, 0.0],  # Grade 6 gaps
        [0.0, 14.0, 0.0, 0.0, 0.0],  # Grade 7 gaps
        [0.0, 12.0, 0.0, 0.0, 0.0],  # Grade 8 gaps
        [0.0, 14.0, 16.0, 0.0, 0.0],  # Grade 9 gaps
        [0.0, 0.0, 0.0, 0.0, 0.0]     # Grade 10 gaps
    ]]),
    
    "all_lengths": torch.tensor([[
        0,  # Grade 0 length
        0,  # Grade 1 length
        # ... grades 2-6 (all 0)
        0,  # Grade 6 length
        2,  # Grade 7 length
        2,  # Grade 8 length
        3,  # Grade 9 length
        1   # Grade 10 length
    ]])
}

# Run inference
with torch.no_grad():
    prediction = model(**inference_input)

print(f"Predicted price for PSA 9 1986 Fleer MJ: ${prediction.item():.2f}")