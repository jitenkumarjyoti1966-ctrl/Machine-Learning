import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

iris = load_iris()

df = pd.DataFrame(iris.data, columns = iris.feature_names)

df["target"] = iris.target

print("--- First Five Rows ---")
print(df.head())

print("\n--- Dataset Information ---")
print(df.info())

print("\n--- Statiscal Summary ---")
print(df.describe())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Correlation Matrix ---")
print(df.corr(numeric_only = True))

plt.figure(figsize = (7, 5))

sns.scatterplot(

    data = df,
    x = "sepal length (cm)", 
    y = "petal length (cm)",
    hue = "target",
    palette = "viridis"

)

plt.title("sepal Length vs Petal Length by class")
plt.show()

plt.figure(figsize = (7, 5))
sns.histplot(df["sepal length (cm)"], kde = True, color = "blue")

plt.title("distribution of Sepal Length")
plt.show()