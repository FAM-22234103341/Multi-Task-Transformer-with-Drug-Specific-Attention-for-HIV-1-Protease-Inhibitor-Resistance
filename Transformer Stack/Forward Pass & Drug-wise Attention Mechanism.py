    def forward(self, x):
        # x shape: (batch_size, seq_len)
        batch_size = x.size(0)
        
        # Add channel dimension and project to d_model
        x = x.unsqueeze(-1)  # (batch_size, seq_len, 1)
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        # Transformer encoding
        encoded = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)
        
        # Drug-specific predictions with attention
        predictions = []
        attentions = []
        
        for i in range(self.num_drugs):
            # Expand drug query for batch
            drug_query = self.drug_queries[i].expand(batch_size, -1, -1)  # (batch_size, 1, d_model)
            
            # Attention between drug query and sequence
            attn_output, attn_weights = self.drug_attention(
                drug_query, encoded, encoded
            )  # attn_output: (batch_size, 1, d_model)
            
            # Predict resistance for this drug
            drug_pred = self.drug_predictors[i](attn_output.squeeze(1))  # (batch_size, 1)
            predictions.append(drug_pred)
            attentions.append(attn_weights)
        
        # Stack predictions
        predictions = torch.cat(predictions, dim=1)  # (batch_size, num_drugs)
        
        return predictions, attentions
