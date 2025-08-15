#1. Import libraries
#import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

#2. Load the dataset
iris = load_iris()
X = iris.data #features
Y = iris.target #labels

#3. Split data into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

#4. Create and train the model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

#5. Make predictions
y_pred = model.predict(X_test)

#6. Evaluate
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))