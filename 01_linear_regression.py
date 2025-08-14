import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dados fictícios
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([3, 4, 2, 5, 6])

# Modelo de regressão
model = LinearRegression()
model.fit(X, y)

# Previsão
y_pred = model.predict(X)

# Visualização
plt.scatter(X, y, color='blue', label='Dados reais')
plt.plot(X, y_pred, color='red', label='Previsão')
plt.legend()
plt.title("Regressão Linear")
plt.xlabel("X")
plt.ylabel("y")
plt.show()
