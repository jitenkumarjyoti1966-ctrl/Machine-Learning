import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_wine

wine_data = load_wine()

wine_df = pd.DataFrame
(
    wine_data.data,
    columns=wine_data.feature_names
)

total_features = len(wine_df.columns)

fig, axes = plt.subplots(4, 4, figsize=(16, 12))
axes = axes.flatten()

for position, feature in enumerate(wine_df.columns):
    sns.boxplot
(
        data=wine_df,
        y=feature,
        color="lightgreen",
        ax=axes[position]
    )
    
    axes[position].set_title
(
        feature,
        fontsize=10,
        fontweight="bold"
    )
    axes[position].set_ylabel("")

for position in range(total_features, len(axes)):
    axes[position].set_visible(False)

fig.suptitle
(
    "Boxplots for Numerical Attributes (Wine Dataset)",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout()
plt.show()
