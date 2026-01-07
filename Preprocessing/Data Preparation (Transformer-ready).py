# Prepare data for transformer - FIXED VERSION
def prepare_transformer_data(sequence_matrix, drug_df, drug_cols):
    # Normalize drug resistance values (log transform for skewed data)
    drug_data = drug_df[drug_cols].copy()
    
    # Log transform to handle skewed resistance values
    drug_data_log = np.log1p(drug_data)
    
    # Standardize each drug separately
    scaler = StandardScaler()
    drug_data_scaled = scaler.fit_transform(drug_data_log)
    
    # Create mask for available drug data
    drug_mask = (~drug_df[drug_cols].isna()).astype(float)
    
    # Fill NaN values with 0 (they will be masked in loss)
    drug_data_filled = np.nan_to_num(drug_data_scaled, nan=0.0)
    
    return drug_data_filled, drug_mask, scaler

# Apply preprocessing
drug_data_scaled, drug_mask, drug_scaler = prepare_transformer_data(
    sequence_matrix, cleaned_df, drug_cols
)

print("Drug data shape:", drug_data_scaled.shape)
print("Drug mask shape:", drug_mask.shape)
print("Available data points:", np.sum(drug_mask))
