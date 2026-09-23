import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

california = fetch_california_housing()

records = pd.DataFrame
(
    california.data,
    columns=california.feature_names
)

records["HousePrice"] = california.target

print("First Five Records:")
print(records.head())

print("\nDataset Shape:")
print(records.shape)

rooms = records[["AveRooms"]]
prices = records["HousePrice"]

x_train, x_test, y_train, y_test = train_test_split(
    rooms,
    prices,
    test_size=0.25,
    random_state=7
)

model = LinearRegression()
model.fit(x_train, y_train)

predictions = model.predict(x_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("\nModel Evaluation")
print("----------------")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

print("\nRegression Equation:")
print(
    f"HousePrice = {model.intercept_:.4f} + "
    f"{model.coef_[0]:.4f} × AveRooms"
)

rooms_input = float
(
    input("\nEnter average number of rooms: ")
)

prediction = model.predict
(
    pd.DataFrame({"AveRooms": [rooms_input]})
)

print(
    f"Predicted House Price: "
    f"${prediction[0] * 100000:,.2f}"
)

plt.figure(figsize=(10, 6))

plt.scatter
(
    x_test["AveRooms"],
    y_test,
    color="green",
    alpha=0.5,
    label="Actual Prices"
)

sorted_test = x_test.sort_values("AveRooms")

plt.plot
(
    sorted_test["AveRooms"],
    model.predict(sorted_test),
    color="red",
    linewidth=2,
    label="Regression Line"
)

plt.xlabel("Average Number of Rooms")
plt.ylabel("House Price")
plt.title("House Price Prediction Using Linear Regression")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
