import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_wine

wine = load_wine()
df = pd.DataFrame(wine.data, columns = wine.feature_names)
plt.figure(figsize = (16, 12))

for i, column in enumerate(df.columns, 1):
    plt.subplot(4, 4, i)
    sns.boxplot(y = df[column], color = 'skyblue')
    plt.title(column, fontsize = 10, fontweight = 'bold')
    plt. ylabel ('')

    for j in range(len(df.columns) + 1, 17):
        plt.subplot(4, 4, j).axis('off')
        plt.suptitle('Boxplots for Nunerical Attributes (Wine Dataset)', fontsize = 14, fontweight = 'bold', y = 1.02)
        plt.tight_layout()

        plt.show()