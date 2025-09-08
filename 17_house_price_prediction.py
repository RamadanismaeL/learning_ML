# 1. Import libraries
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 2. Load dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

print("Dataset shape: ", df.shape)
print(df.head())

# 3. Define features (x) and target (y)
x = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# 4. Split into train/test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 5. Train model
model = LinearRegression()
model.fit(x_train, y_train)

# 6. Predictions
y_pred = model.predict(x_test)

# 7. Evaluation
print("MSE: ", mean_squared_error(y_test, y_pred))
print("R² score: ", r2_score(y_test, y_pred))

# 8. Visualization
plt.scatter(y_test, y_pred, spha=0.5)
plt.xlabel("True Prices")
plt.ylabel("Predicteed Prices")
plt.title("House Price Prediction")
plt.show()