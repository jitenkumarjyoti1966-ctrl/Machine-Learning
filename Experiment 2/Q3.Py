import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns 
from sklearn.datasets import load_wine

wine = load_wine()
df = pd.DataFrame(wine.data, columns = wine.feature_names)

corr_matrix = df.corr()

plt.figure(figsize = (12, 10))
sns.heatmap(
    corr_matrix,
    annot = True,
    fmt = '.2f',
    cmap = 'coolwarm',
    vmin = -1,
    vmax = 1,
    linewidth = 0.5
)
plt.title('Correlation Heatmap -Wine Dataset', fontsize = 14, fontweight = 'bold', pad =15)
plt.tight_layout()

plt.savefig('winr_correlation_heatmap.png', dpi = 300)
plt.close('all')

corr_no_diag = corr_matrix.copy()
np.fill_diagonal(corr_no_diag.values, np.nan)

sorted_corr = corr_no_diag.unstack().dropna().sort_values(ascending = False)
feat1, feat2 = sorted_corr.index[0]
corr_val = sorted_corr . iloc[0]

print("=" * 50)
print("Correlation Analysis Results")
print("=" * 50)
print(f"Strongest Positive Correlation: {feat1} <--> {feat2}")
print(f"Pearson Correlation Coefficient (r): {corr_val:.4f}")
print("=" * 50)
print("Saved heatmap image to: wine_correlation_heatmap.png")
