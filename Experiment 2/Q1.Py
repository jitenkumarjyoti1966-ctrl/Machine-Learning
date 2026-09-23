import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

sns.set_theme(style="ticks")

wine_data = load_wine()

wine_df = pd.DataFrame
(
    wine_data.data,
    columns=wine_data.feature_names
)

wine_df.insert(0, "Class", wine_data.target)

wine_df["Wine Type"] = wine_df["Class"].replace
(
    dict(enumerate(wine_data.target_names))
)

print("=== Dataset Preview ===")
print(wine_df.iloc[:5])

print("\n=== Dataset Shape ===")
rows, columns = wine_df.shape
print("Total Rows:", rows)
print("Total Columns:", columns)

print("\n=== Dataset Information ===")
wine_df.info()

print("\n=== Checking Missing Data ===")
print(wine_df.isna().sum())

print("\n=== Statistical Summary ===")
stats = wine_df.describe().T
print(stats.loc[:, ["mean", "std", "min", "50%", "max"]])

print("\n=== Wine Type Frequency ===")
print(wine_df["Wine Type"].value_counts())

plt.figure(figsize=(11, 7))

numeric_features = wine_df.select_dtypes(
    include=["int64", "float64"]
)

sns.heatmap
(
    numeric_features.corr(),
    annot=True,
    cmap="RdBu_r",
    center=0,
    fmt=".2f"
)

plt.title("Wine Feature Correlation Matrix")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))

sns.countplot
(
    x="Wine Type",
    data=wine_df,
    hue="Wine Type",
    palette="Set3",
    legend=False
)

plt.title("Distribution of Wine Classes")
plt.xlabel("Wine Class")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.show()

figure, chart = plt.subplots(1, 2, figsize=(14, 5))

sns.boxplot
(
    data=wine_df,
    x="Wine Type",
    y="alcohol",
    hue="Wine Type",
    palette="Set3",
    legend=False,
    ax=chart[0]
)

chart[0].set_title("Alcohol Content Across Wine Classes")
chart[0].set_xlabel("Wine Class")
chart[0].set_ylabel("Alcohol")

sns.boxplot
(
    data=wine_df,
    x="Wine Type",
    y="flavanoids",
    hue="Wine Type",
    palette="Set3",
    legend=False,
    ax=chart[1]
)

chart[1].set_title("Flavanoid Content Across Wine Classes")
chart[1].set_xlabel("Wine Class")
chart[1].set_ylabel("Flavanoids")

plt.tight_layout()
plt.show()
