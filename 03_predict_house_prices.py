# We'll predict house prices using the California housing dataset

# Project: Predict House Prices (Regression)

#1. Import libraries
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#2. Load dataset
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
Y = housing.target # Median house value in $100,000s

#3. Split into train/test
X_train, X_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#4. Create and train model
model = LinearRegression()
model.fit(X_train, Y_train)

#5. Predictions
y_pred = model.predict(X_test)

#6. Evaluation
print("Mean Squared Error: ", mean_squared_error(y_test, y_pred))
print("R² Score: ", r2_score(y_test, y_pred))

#7. Example: predict for the first 5 houses in the test set
print("\nPredictions for first 5 houses:", y_pred[:5])
print("Actual values: ", y_test[:5])