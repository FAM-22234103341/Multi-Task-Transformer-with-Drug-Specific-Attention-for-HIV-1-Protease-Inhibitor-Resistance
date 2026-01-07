# Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                sequences = batch['sequence'].to(device)
                drug_resistance = batch['drug_resistance'].to(device)
                drug_mask = batch['drug_mask'].to(device)
                
                predictions, _ = model(sequences)
                loss = criterion(predictions, drug_resistance, drug_mask)
                val_loss += loss.item()
