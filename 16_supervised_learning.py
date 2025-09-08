from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load daaset
data = load_boston()
x, y = data.data, data.target

# Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train
model = LinearRegression()
model.fit(x_train, y_train)

# Predict
y_pred = model.predict(x_test)
print("MSE: ", mean_squared_error(y_test, y_pred))