import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

housing = fetch_california_housing()

df = pd.DataFrame(
    housing.data,
    columns=housing.feature_names
)

df["Price"] = housing.target

X = df[["MedInc", "AveRooms"]]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X_test)
linear_r2 = r2_score(y_test, linear_pred)

poly = PolynomialFeatures(degree=3)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

poly_pred = poly_model.predict(X_test_poly)
poly_r2 = r2_score(y_test, poly_pred)

print("Linear Regression R2 Score:", linear_r2)
print("Polynomial Regression R2 Score:", poly_r2)

print("\nComparison")
print("Linear Regression     :", linear_r2)
print("Polynomial Regression :", poly_r2)

if poly_r2 > linear_r2:
    print("\nPolynomial Regression has a higher R2 score.")
else:
    print("\nLinear Regression has a higher R2 score.")
