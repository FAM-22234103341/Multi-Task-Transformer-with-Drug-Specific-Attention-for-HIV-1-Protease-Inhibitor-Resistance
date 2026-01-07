# 2. Missing values heatmap
missing_data = cleaned_df[drug_cols].isna()
sns.heatmap(missing_data, cbar=True, ax=axes[0,1])
axes[0,1].set_title('Missing Values in Drug Resistance Data')
