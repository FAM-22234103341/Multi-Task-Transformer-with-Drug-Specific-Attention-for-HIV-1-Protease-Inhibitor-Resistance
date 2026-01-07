class ProteaseTransformer(nn.Module):
    def __init__(self, seq_len=99, d_model=64, nhead=8, num_layers=3, num_drugs=8, dropout=0.1):
        super(ProteaseTransformer, self).__init__()
        
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_drugs = num_drugs
        
        # Input projection
        self.input_projection = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, seq_len)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Drug-specific prediction networks
        self.drug_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1)
            ) for _ in range(num_drugs)
        ])
        
        # Global pooling for each drug
        self.drug_attention = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=nhead, 
            dropout=dropout, 
            batch_first=True
        )
        
        # Learnable drug queries
        self.drug_queries = nn.Parameter(torch.randn(num_drugs, 1, d_model))
