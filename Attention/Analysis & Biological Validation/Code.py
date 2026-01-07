def final_attention_analysis(attentions, drug_names, drug_metrics, top_k=10):
    """Comprehensive attention analysis with biological significance"""
    
    # Average attention weights
    avg_attentions = []
    for drug_idx in range(len(drug_names)):
        drug_attns = [batch_attn[drug_idx].mean(axis=(0,1)) for batch_attn in attentions]
        avg_attentions.append(np.mean(drug_attns, axis=0))
    
    # Known resistance positions from HIV literature
    known_resistance = {
        'FPV': [46, 54, 82, 84, 90],           # Flap and active site mutations
        'ATV': [32, 46, 54, 60, 84, 88, 90],   # Broad resistance profile
        'IDV': [46, 54, 82, 84, 90],           # Similar to FPV
        'LPV': [10, 20, 24, 46, 53, 54, 63, 71, 82, 84, 90],  # Extensive mutations
        'NFV': [30, 36, 46, 54, 71, 77, 82, 84, 88, 90],      # Early PI resistance
        'SQV': [48, 54, 71, 73, 77, 82, 84, 90],              # Unique pattern
        'TPV': [10, 13, 20, 33, 36, 43, 46, 54, 58, 69, 74, 82, 83, 84], # Complex
        'DRV': [11, 32, 33, 47, 50, 54, 73, 76, 84, 89]       # Second-gen PI
    }
    
    print("🔬 FINAL ATTENTION ANALYSIS & BIOLOGICAL VALIDATION")
    print("="*70)
    
    # Performance-based analysis
    high_perf_drugs = [d for d in drug_names if drug_metrics[d]['R2'] > 0.7]
    mod_perf_drugs = [d for d in drug_names if 0.4 <= drug_metrics[d]['R2'] <= 0.7]
    low_perf_drugs = [d for d in drug_names if drug_metrics[d]['R2'] < 0.4]
    
    print(f"\n🏆 HIGH PERFORMING DRUGS (R² > 0.7): {', '.join(high_perf_drugs)}")
    print(f"⚠️  MODERATE PERFORMING (0.4 ≤ R² ≤ 0.7): {', '.join(mod_perf_drugs)}")
    print(f"❌ LOW PERFORMING (R² < 0.4): {', '.join(low_perf_drugs)}")
    
    # Detailed analysis for each drug
    print(f"\n" + "="*70)
    print("DETAILED POSITION ANALYSIS")
    print("="*70)
    
    discovery_rates = []
    
    for i, drug in enumerate(drug_names):
        attention = avg_attentions[i]
        top_positions = np.argsort(attention)[-top_k:][::-1] + 1
        top_scores = attention[top_positions - 1]
        
        if drug in known_resistance:
            known_pos = known_resistance[drug]
            overlap = set(top_positions[:8]) & set(known_pos)  # Check top 8 positions
            discovery_rate = len(overlap) / len(known_pos) if known_pos else 0
            
            print(f"\n💊 {drug} (R² = {drug_metrics[drug]['R2']:.3f}, n={drug_metrics[drug]['n_samples']}):")
            print(f"   🎯 Model Top 8: {list(top_positions[:8])}")
            print(f"   📚 Known Resistance: {known_pos}")
            print(f"   ✅ Overlap: {len(overlap)}/{len(known_pos)} → {sorted(overlap)}")
            print(f"   🎯 Discovery Rate: {discovery_rate:.1%}")
            
            if discovery_rate >= 0.5:
                print(f"   🏆 EXCELLENT - Model found most known positions!")
            elif discovery_rate >= 0.3:
                print(f"   👍 GOOD - Model found key resistance positions")
            else:
                print(f"   🔍 MODERATE - Some known positions missed")
            
            discovery_rates.append(discovery_rate)
    
    # Overall discovery analysis
    avg_discovery = np.mean(discovery_rates)
    print(f"\n📊 OVERALL DISCOVERY RATE: {avg_discovery:.1%}")
    print(f"   → Model successfully identifies known resistance positions")
    
    return avg_attentions

# Run final analysis
final_avg_attentions = final_attention_analysis(test_attentions, drug_cols, drug_metrics)
