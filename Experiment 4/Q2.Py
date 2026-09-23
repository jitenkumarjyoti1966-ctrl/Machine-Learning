import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

california_data = fetch_california_housing()

house_df = pd.DataFrame
(
    california_data.data,
    columns=california_data.feature_names
)

house_df["House_Value"] = california_data.target

print("Sample Housing Data:")
print(house_df.head(5))

print("\nShape of Housing Dataset:")
print("Rows:", house_df.shape[0])
print("Columns:", house_df.shape[1])

features = house_df.loc[:, ["MedInc", "AveRooms"]]
target = house_df.loc[:, "House_Value"]

train_features, test_features, train_target, test_target = train_test_split
(
    features,
    target,
    test_size=0.25,
    random_state=15
)

regression = LinearRegression()
regression.fit(train_features, train_target)

predictions = regression.predict(test_features)

error1 = mean_absolute_error(test_target, predictions)
error2 = mean_squared_error(test_target, predictions)
error3 = np.sqrt(error2)
accuracy = r2_score(test_target, predictions)

print("\nPerformance Results")
print("Mean Absolute Error :", error1)
print("Mean Squared Error  :", error2)
print("Root Mean Squared Error:", error3)
print("R2 Score            :", accuracy)

weights = regression.coef_
constant = regression.intercept_

print("\nRegression Coefficients:")
print("MedInc   :", weights[0])
print("AveRooms :", weights[1])
print("Intercept:", constant)

print("\nRegression Equation:")
print(
    f"House_Value = {constant:.4f} "
    f"+ {weights[0]:.4f} × MedInc "
    f"+ {weights[1]:.4f} × AveRooms"
)

income_value = float(input("\nEnter median income: "))
room_value = float(input("Enter average number of rooms: "))

user_house = pd.DataFrame(
    [[income_value, room_value]],
    columns=["MedInc", "AveRooms"]
)

estimated_price = regression.predict(user_house)

print(f"\nPredicted House Price: "f"${estimated_price[0] * 100000:,.2f}")

plt.figure(figsize=(9, 6))

plt.scatter
(
    test_target,
    predictions,
    color="darkgreen",
    alpha=0.55,
    edgecolors="black"
)

plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted Values - Multiple Linear Regression")
plt.grid(alpha=0.3)

plt.show()
