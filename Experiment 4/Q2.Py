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

X = df[["MedInc", "AveRooms"]]
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
print("MAE  :", mae)
print("MSE  :", mse)
print("RMSE :", rmse)
print("R2   :", r2)

print("\nRegression Coefficients:")
print("MedInc    :", model.coef_[0])
print("AveRooms  :", model.coef_[1])
print("Intercept :", model.intercept_)

print("\nRegression Equation:")
print(
    "Price =",
    model.intercept_,
    "+",
    model.coef_[0],
    "* MedInc +",
    model.coef_[1],
    "* AveRooms"
)

income = float(input("\nEnter median income: "))
rooms = float(input("Enter average number of rooms: "))

new_house = pd.DataFrame({
    "MedInc": [income],
    "AveRooms": [rooms]
})

prediction = model.predict(new_house)

print(f"\nPredicted house price: "f"${prediction[0] * 100000:,.2f}")

plt.scatter(
    y_test,
    y_pred,
    color="blue",
    alpha=0.5
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Multiple Linear Regression")

plt.show()
