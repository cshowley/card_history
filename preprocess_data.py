def preprocess_for_full_grade(raw_df, max_seq_len=12, test_size=0.15):
    """
    Creates training data that includes ALL grade histories
    
    Critical: For each sale, we inject ALL available grade histories
    """
    df = raw_df.copy()
    
    # Rename columns for clarity
    df = df.rename(columns={
        'spec_id': 'card_id',
        'market_transaction.date': 'sale_date',
        'market_transaction.price': 'price',
        'market_transaction.attributes.gradeNumber': 'grade'
    })
    
    # Filter and clean
    df = df[df['price'] > 0]
    df['sale_date'] = pd.to_datetime(df['sale_date'])
    df = df.sort_values(['card_id', 'grade', 'sale_date'])
    
    # Create card ID mapping
    unique_cards = sorted(df['card_id'].unique())
    card_to_idx = {card: i+1 for i, card in enumerate(unique_cards)}
    
    # Calculate time gaps per card-grade
    df['prev_date'] = df.groupby(['card_id', 'grade'])['sale_date'].shift(1)
    df['days_gap'] = (df['sale_date'] - df['prev_date']).dt.days
    df['days_gap'] = df['days_gap'].fillna(0).clip(lower=0)
    
    # Build training sequences
    all_data = []
    
    # Process each card
    for card_id, card_group in df.groupby('card_id'):
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
                # Build ALL grade histories as of this sale date
                current_date = dates[i]
                all_prices = np.zeros((11, max_seq_len))
                all_gaps = np.zeros((11, max_seq_len))
                all_lengths = np.zeros(11, dtype=int)
                
                # For each possible grade
                for g in range(0, 11):
                    if g in grade_histories:
                        g_sales = grade_histories[g]
                        # Filter sales up to current_date
                        mask = g_sales['dates'] <= current_date
                        g_prices = g_sales['prices'][mask]
                        g_gaps = g_sales['gaps'][mask]
                        
                        # Store in appropriate position
                        all_lengths[g] = len(g_prices)
                        all_prices[g, :len(g_prices)] = g_prices
                        all_gaps[g, :len(g_gaps)] = g_gaps
                
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
    
    # Split data
    full_df = pd.DataFrame(all_data)
    train_df, val_df = train_test_split(
        full_df, 
        test_size=test_size, 
        random_state=42,
        stratify=full_df['card_idx']
    )
    
    # Create Dataset objects
    train_dataset = FullGradeDataset(train_df, max_seq_len)
    val_dataset = FullGradeDataset(val_df, max_seq_len)
    
    return train_dataset, val_dataset, card_to_idx

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