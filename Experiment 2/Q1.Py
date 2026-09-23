import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.datasets import load_wine

sns.set_theme(style = "whitegrid")
wine = load_wine()

df = pd.DataFrame(wine.data, columns = wine.feature_names)
df['target'] = wine.target

df['class_name'] = df['target'].map(lambda x: wine.target_names[x])

print("=== First Five Records ===")
print(df.head())

print("\n=== Dataset Dimensions ===")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n=== Data Types & Non-Null Counts ===")
print(df.info())

print("\n=== Missing Values ===")
print(df.isnull().sum())

print("\n=== Summary Statistics (Numerical Features) ===")
print(df.describe().T[['mean', 'std', 'min', '50%', 'max']])

print("\n=== Target Class Distribution ===")
print(df['class_name'].value_counts())

plt.figure(figsize = (12, 8))

numeric_df = df.drop(columns = ['class_name'])

sns.heatmap(numeric_df.corr(), annot = True, cmap = "coolwarm", fmt = ".2f", linewidths = 0.5)

plt.title("feature Correlation Matrix")
plt.tight_layout()
plt.show()

plt.figure(figsize = (6, 4))
sns.countplot(data = df, x = 'class_name', palette = 'Set2')

plt.title("Wine Class Count Distribution")
plt.xlabel("Wine Class")

plt.ylabel("Count")
plt.show()

fig, axes = plt.subplots(1, 2, figsize = (14, 5))

sns.boxplot(ax = axes[0], data = df, x = 'class_name', y = 'alcohol', palette = 'Set2')
axes[0].set_title("Alcohol Level by Wine Class")

sns.boxplot(ax = axes[1], data = df, x = 'class_name', y = 'flavanoids', palette = 'Set2')
axes[1].set_title("Flavanoid Content by Wine Class")

plt.tight_layout()
plt.show()