import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

housing = fetch_california_housing()

df = pd.DataFrame(
    housing.data,
    columns=housing.feature_names
)

df["Price"] = housing.target

print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

X = df[["AveRooms"]]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("------------------------")
print("MAE  :", mae)
print("MSE  :", mse)
print("RMSE :", rmse)
print("R2   :", r2)

print("\nLinear Regression Equation:")
print(
    "Price =",
    model.coef_[0],
    "* AveRooms +",
    model.intercept_
)

rooms = float(input("\nEnter average number of rooms: "))

new_prediction = model.predict(
    pd.DataFrame({"AveRooms": [rooms]})
)

print(
    f"Predicted house price: "
    f"${new_prediction[0] * 100000:,.2f}"
)

plt.scatter(
    X_test,
    y_test,
    color="blue",
    alpha=0.5,
    label="Actual prices"
)

plt.plot(
    X_test,
    y_pred,
    color="red",
    linewidth=2,
    label="Regression line"
)

plt.xlabel("Average Number of Rooms")
plt.ylabel("House Price")
plt.title("House Price Prediction Using One Feature")

plt.legend()
plt.show()
