import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

np.random.seed(42)
n_samples = 100

data = {
    "Age": np.random.randint(22, 660, size = n_samples).astype(float),
    "Salary": np.random.normal(65000, 15000, size = n_samples).round(-2),
    "Department": np.random.choice(["HR", "Engineering", "Marketing", "Sales"], size = n_samples),
    "Years_Of_Experience": np.random.randint(1, 35, size = n_samples).astype(float),
}

df = pd.DataFrame(data)

missing_rate = 0.15

for col in df.columns:
    mask = np.random.rand(len(df)) < missing_rate
    df.loc[mask, col] = np.nan

    print("--- Raw Dataset with Missing Values (First Five Rows) ---")
    print(df.head())

    print("\nMissing values summary:")
    print(df.isnull().sum())

    num_features = ["Age", "Salary", "Years_Of_Experience"]
    cat_features = ["Department"]

    num_pipeline = Pipeline(
        [
        ("imputer", SimpleImputer(strategy = "median")),
        ("scaler", MinMaxScaler()),
        ]
    )

    cat_pipeline = Pipeline (
        [
            ("imputer", SimpleImputer(strategy = "most_frequent")),
            ("encoder", OneHotEncoder(sparse_output = False, drop = "first")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers = [
            ("num", num_pipeline, num_features),
            ("cat", cat_pipeline, cat_features)
        ]
    )

    processed_array = preprocessor.fit_transform(df)

    encoded_cat_cols = (
        preprocessor.named_transformers_["cat"]["encoder"].get_feature_names_out(cat_features).tolist()
    )
    feature_names = num_features + encoded_cat_cols

    df_processed = pd.DataFrame(processed_array, columns = feature_names)

    print("\n--- Preprocessed Clean Dataset (First Five Rows) ---")
    print(df_processed.head())
