    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            sequences = batch['sequence'].to(device)
            drug_resistance = batch['drug_resistance'].to(device)
            drug_mask = batch['drug_mask'].to(device)
            
            optimizer.zero_grad()
            predictions, _ = model(sequences)
            loss = criterion(predictions, drug_resistance, drug_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
