def evaluate_model(model, test_loader):
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_masks = []
    all_attentions = []
    
    with torch.no_grad():
        for batch in test_loader:
            sequences = batch['sequence'].to(device)
            drug_resistance = batch['drug_resistance'].to(device)
            drug_mask = batch['drug_mask'].to(device)
            
            predictions, attentions = model(sequences)
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(drug_resistance.cpu().numpy())
            all_masks.append(drug_mask.cpu().numpy())
            all_attentions.append([attn.cpu().numpy() for attn in attentions])
    
    # Concatenate results
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)
    masks = np.concatenate(all_masks)
    
    # Calculate metrics per drug
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    drug_metrics = {}
    for i, drug in enumerate(drug_cols):
        drug_mask = masks[:, i].astype(bool)
        if np.sum(drug_mask) > 0:
            drug_pred = predictions[drug_mask, i]
            drug_target = targets[drug_mask, i]
            
            mse = mean_squared_error(drug_target, drug_pred)
            mae = mean_absolute_error(drug_target, drug_pred)
            r2 = r2_score(drug_target, drug_pred)
            
            drug_metrics[drug] = {
                'MSE': mse,
                'MAE': mae, 
                'R2': r2,
                'n_samples': np.sum(drug_mask)
            }
    
    return drug_metrics, predictions, targets, masks, all_attentions

# Evaluate model
print("Evaluating model...")
drug_metrics, test_predictions, test_targets, test_masks, test_attentions = evaluate_model(model, test_loader)

# Print results
print("\n" + "="*60)
print("PER-DRUG PERFORMANCE METRICS")
print("="*60)
for drug, metrics in drug_metrics.items():
    print(f"{drug:4s}: MSE = {metrics['MSE']:.4f}, MAE = {metrics['MAE']:.4f}, R² = {metrics['R2']:.4f}, n = {metrics['n_samples']}")

# Calculate overall metrics
overall_mse = np.mean([m['MSE'] for m in drug_metrics.values()])
overall_mae = np.mean([m['MAE'] for m in drug_metrics.values()])
overall_r2 = np.mean([m['R2'] for m in drug_metrics.values()])

print(f"\nOVERALL: MSE = {overall_mse:.4f}, MAE = {overall_mae:.4f}, R² = {overall_r2:.4f}")
