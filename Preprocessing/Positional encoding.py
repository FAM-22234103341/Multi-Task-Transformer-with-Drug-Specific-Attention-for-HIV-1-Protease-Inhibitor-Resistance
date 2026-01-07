class ProteaseDataset(Dataset):
    def __init__(self, sequences, drug_resistance, drug_mask):
        self.sequences = sequences  # Shape: (n_samples, seq_len)
        self.drug_resistance = drug_resistance  # Shape: (n_samples, n_drugs)
        self.drug_mask = drug_mask  # Shape: (n_samples, n_drugs) - 1 if available, 0 if missing
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'sequence': torch.FloatTensor(self.sequences[idx]),
            'drug_resistance': torch.FloatTensor(self.drug_resistance[idx]),
            'drug_mask': torch.FloatTensor(self.drug_mask[idx])
        }
