import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_wine

wine_data = load_wine()

wine_df = pd.DataFrame
(
    wine_data.data,
    columns=wine_data.feature_names
)

correlations = wine_df.corr(method="pearson")

fig, ax = plt.subplots(figsize=(12, 10))

sns.heatmap
(
    correlations,
    annot=True,
    fmt=".2f",
    cmap="RdYlBu_r",
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    ax=ax
)

ax.set_title
(
    "Wine Dataset - Feature Correlation",
    fontsize=14,
    fontweight="bold",
    pad=15
)

plt.tight_layout()
plt.savefig("wine_features_correlation.png", dpi=300)
plt.close(fig)

temp_corr = correlations.mask
(
    np.eye(correlations.shape[0], dtype=bool)
)

correlation_pairs = (
    temp_corr.stack()
    .sort_values(ascending=False)
)

feature_a, feature_b = correlation_pairs.index[0]
highest_value = correlation_pairs.iloc[0]

print("-" * 50)
print("Correlation Analysis")
print("-" * 50)
print(f"Highest Positive Correlation: {feature_a} <--> {feature_b}")
print(f"Pearson Correlation (r): {highest_value:.4f}")
print("-" * 50)
print("Heatmap saved as: wine_features_correlation.png")
