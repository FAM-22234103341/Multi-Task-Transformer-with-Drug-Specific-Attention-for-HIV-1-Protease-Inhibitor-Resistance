# 1. Drug resistance distribution
drug_data = cleaned_df[drug_cols].melt(var_name='Drug', value_name='Resistance')
sns.boxplot(data=drug_data, x='Drug', y='Resistance', ax=axes[0,0])
axes[0,0].set_title('Drug Resistance Distribution')
axes[0,0].tick_params(axis='x', rotation=45)
