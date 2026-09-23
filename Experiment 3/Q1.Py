import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

np.random.seed(25)
total_records = 80

employee_data = {
    "Age": np.random.randint(21, 60, total_records).astype(float),
    "Salary": np.random.normal(55000, 12000, total_records).round(-2),
    "Department": np.random.choice(
        ["Finance", "IT", "Operations", "Design"],
        total_records
    ),
    "Years_Of_Experience": np.random.randint(0, 30, total_records).astype(float)
}

employee_df = pd.DataFrame(employee_data)

null_percentage = 0.10

for field in employee_df.columns:
    missing = np.random.random(len(employee_df)) < null_percentage
    employee_df.loc[missing, field] = np.nan

print("--- Original Data with Missing Entries ---")
print(employee_df.head())

print("\nMissing Entry Count:")
print(employee_df.isna().sum())

numerical_columns = ["Age", "Salary", "Years_Of_Experience"]
categorical_columns = ["Department"]

numerical_steps = Pipeline(
[
    ("fill", SimpleImputer(strategy="mean")),
    ("normalize", StandardScaler())
]
)

categorical_steps = Pipeline(
[
    ("fill", SimpleImputer(strategy="most_frequent")),
    ("convert", OneHotEncoder(sparse_output=False, drop="first"))
]
)

data_transformer = ColumnTransformer(
[
    ("numbers", numerical_steps, numerical_columns),
    ("categories", categorical_steps, categorical_columns)
]
)

clean_data = data_transformer.fit_transform(employee_df)

category_names = (
    data_transformer
    .named_transformers_["categories"]["convert"]
    .get_feature_names_out(categorical_columns)
    .tolist()
)

all_columns = numerical_columns + category_names

clean_df = pd.DataFrame(clean_data, columns=all_columns)

print("\n--- Processed Dataset ---")
print(clean_df.head())

