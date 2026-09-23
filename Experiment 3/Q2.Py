import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

np.random.seed(25)
records = 80

employee_data = {
    "Age": np.random.randint(20, 60, records).astype(float),
    "Salary": np.random.normal(58000, 13000, records).round(-2),
    "Department": np.random.choice(
        ["Finance", "IT", "Operations", "Design"],
        records
    ),
    "Years_Of_Experience": np.random.randint(1, 30, records).astype(float)
}

employee_df = pd.DataFrame(employee_data)

null_rate = 0.10

for feature in employee_df.columns:
    missing_values = np.random.random(records) < null_rate
    employee_df.loc[missing_values, feature] = np.nan

print("--- Raw Dataset with Missing Values ---")
print(employee_df.head())

print("\nMissing Values Summary:")
print(employee_df.isna().sum())

numeric_columns = ["Age", "Salary", "Years_Of_Experience"]
category_columns = ["Department"]

numeric_process = Pipeline([
    ("replace", SimpleImputer(strategy="median")),
    ("scale", MinMaxScaler())
])

category_process = Pipeline(
[
    ("replace", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(sparse_output=False, drop="first"))
]
)

transformer = ColumnTransformer(
[
    ("numeric_data", numeric_process, numeric_columns),
    ("category_data", category_process, category_columns)
]
)

transformed_data = transformer.fit_transform(employee_df)

encoded_columns = (
    transformer
    .named_transformers_["category_data"]["encode"]
    .get_feature_names_out(category_columns)
    .tolist()
)

column_names = numeric_columns + encoded_columns

processed_df = pd.DataFrame(
    transformed_data,
    columns=column_names
)

print("\n--- Preprocessed Clean Dataset ---")
print(processed_df.head())
