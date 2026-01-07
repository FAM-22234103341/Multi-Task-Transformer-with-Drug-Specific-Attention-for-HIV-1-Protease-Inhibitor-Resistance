# Data cleaning function
def clean_dataset(df):
    # Create a copy
    data = df.copy()
    
    # Handle drug resistance values - replace non-numeric with NaN
    for col in drug_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Handle mutation columns (P1-P99)
    mutation_cols = [f'P{i}' for i in range(1, 100)]
    
    # Encode mutations: '-' = wild type (0), anything else = mutation (1), NaN = wild type (0)
    for col in mutation_cols:
        data[col] = data[col].apply(lambda x: 0 if x == '-' else (1 if pd.notna(x) and x != '-' else 0))
    
    # Create sequence matrix
    sequence_data = data[mutation_cols].values
    
    # Handle missing drug values - we'll use them for multi-task learning
    # For now, keep them as NaN and handle in loss function
    
    return data, sequence_data, mutation_cols

cleaned_df, sequence_matrix, mutation_cols = clean_dataset(df)

print("Sequence matrix shape:", sequence_matrix.shape)
print("Mutation pattern example (first 5 positions):", sequence_matrix[0, :5])
print("Number of sequences with mutations:", np.sum(sequence_matrix > 0))
