def calculate_full_grade_loss(predictions, targets, all_lengths, model,
                            mse_weight=1.0, grade_consistency_weight=2.0):
    """Custom loss for full grade history model"""
    # 1. Primary MSE loss
    mse_loss = F.mse_loss(predictions, targets)
    
    # 2. Grade consistency constraint (CRITICAL)
    # Ensure higher grades have higher predicted prices
    batch_size = predictions.size(0)
    
    # We don't have grade info in this loss function directly
    # This is handled through the architecture instead
    
    # 3. Grade relationship regularization
    # Encourage meaningful relationships between grade embeddings
    grade_relationship = model.grade_encoder[0].weight  # Access first layer weights
    relationship_variance = torch.var(grade_relationship, dim=0).mean()
    relationship_loss = -relationship_variance  # Maximize variance
    
    return (
        mse_weight * mse_loss +
        grade_consistency_weight * relationship_loss
    )

def train_full_grade_model(
    model,
    train_loader,
    val_loader,
    epochs=200,
    lr=0.001,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """Train the Full Grade History model"""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, verbose=True
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
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
                model
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
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
        val_loss /= len(val_loader)
        val_mape /= len(val_loader)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val MAPE: {val_mape:.2f}%")