# Model initialization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = ProteaseTransformer(
    seq_len=99,
    d_model=64,
    nhead=8,
    num_layers=3,
    num_drugs=len(drug_cols),
    dropout=0.1
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
