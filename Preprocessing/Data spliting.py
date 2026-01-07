# Split data function - ENSURING NUMPY ARRAYS
def train_val_test_split(sequences, drug_data, drug_mask, train_ratio=0.7, val_ratio=0.15, random_state=42):
    n_samples = len(sequences)
    np.random.seed(random_state)
    indices = np.random.permutation(n_samples)
    
    train_size = int(train_ratio * n_samples)
    val_size = int(val_ratio * n_samples)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    return train_indices, val_indices, test_indices
