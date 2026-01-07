# Convert to numpy arrays to avoid pandas indexing issues
sequence_matrix_np = np.array(sequence_matrix)
drug_data_scaled_np = np.array(drug_data_scaled)
drug_mask_np = np.array(drug_mask)

print("Converted to numpy arrays:")
print(f"Sequence matrix: {sequence_matrix_np.shape}")
print(f"Drug data: {drug_data_scaled_np.shape}")
print(f"Drug mask: {drug_mask_np.shape}")

# Create datasets - USING NUMPY ARRAYS
train_idx, val_idx, test_idx = train_val_test_split(
    sequence_matrix_np, drug_data_scaled_np, drug_mask_np
)

print(f"Train indices: {len(train_idx)}, Val indices: {len(val_idx)}, Test indices: {len(test_idx)}")

train_dataset = ProteaseDataset(
    sequence_matrix_np[train_idx],
    drug_data_scaled_np[train_idx], 
    drug_mask_np[train_idx]
)

val_dataset = ProteaseDataset(
    sequence_matrix_np[val_idx],
    drug_data_scaled_np[val_idx],
    drug_mask_np[val_idx]
)

test_dataset = ProteaseDataset(
    sequence_matrix_np[test_idx],
    drug_data_scaled_np[test_idx],
    drug_mask_np[test_idx]
)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Data loaders - DEFINE THEM FIRST
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("Data loaders created successfully!")

# Verify batch shapes
sample_batch = next(iter(train_loader))
print(f"\nSample batch shapes:")
print(f"Sequences: {sample_batch['sequence'].shape}")
print(f"Drug resistance: {sample_batch['drug_resistance'].shape}")
print(f"Drug mask: {sample_batch['drug_mask'].shape}")
