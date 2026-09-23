import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

california = fetch_california_housing()

data = pd.DataFrame
(
    california.data,
    columns=california.feature_names
)

data["House_Value"] = california.target

features = data[["MedInc", "AveRooms"]]
target = data["House_Value"]

x_train, x_test, y_train, y_test = train_test_split
(
    features,
    target,
    test_size=0.25,
    random_state=10
)

model_linear = LinearRegression()
model_linear.fit(x_train, y_train)

prediction_linear = model_linear.predict(x_test)
score_linear = r2_score(y_test, prediction_linear)

transformer = PolynomialFeatures
(
    degree=3,
    include_bias=True
)

train_poly = transformer.fit_transform(x_train)
test_poly = transformer.transform(x_test)

model_poly = LinearRegression()
model_poly.fit(train_poly, y_train)

prediction_poly = model_poly.predict(test_poly)
score_poly = r2_score(y_test, prediction_poly)

print("Linear Model R2:", score_linear)
print("Polynomial Model R2:", score_poly)

print("\nModel Performance Comparison")
print("Linear Model     :", score_linear)
print("Polynomial Model :", score_poly)

if score_linear >= score_poly:
    print("\nLinear Regression has a higher R2 score.")
else:
    print("\nPolynomial Regression has a higher R2 score.")
