# 4. Correlation between drugs
drug_corr = cleaned_df[drug_cols].corr()
sns.heatmap(drug_corr, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1,0])
axes[1,0].set_title('Correlation Between Drugs')
