import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#1. Load dataset (California housing prices)
data = fetch_california_housing(as_frame=True)
X = data.data[["MedInc"]] # Feature: Median income
Y = data.target # Target: House price

#2. Split dataset into train and test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#3. Train model
model = LinearRegression()
model.fit(x_train, y_train)

#4. Predictions
y_pred = model.predict(x_test)

#5. Evaluate
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

#6. Visualization
plt.scatter(x_test, y_test, color="blue", label="Actual prices")
plt.scatter(x_test, y_pred, color="red", label="Predicted prices")
plt.xlabel("Median Income")
plt.ylabel("House Price")
plt.legend()
plt.show()