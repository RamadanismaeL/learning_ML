#Project: Titanic Survival Prediction (Classification)

#1. Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

#2. Load dataset (public Titanic dataset from CSV)
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

#3. Basic preprocessing
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1}) # convert gender to number
#df['Age'].fillna(df['Age'].median(), inplace=True) #fill missing ages with median
df['Age'] = df['Age'].fillna(df['Age'].median())
#df['Embarked'].fillna('S', inplace=True)
df['Embarked'] = df['Embarked'].fillna('S')
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

#Features & Target
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
Y = df['Survived']

#4. Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

#5. Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)

#6. Predictions
Y_pred = model.predict(X_test)

#7. Evaluation
print("Accuracy: ", accuracy_score(Y_test, Y_pred))
print("\nClassification Report:\n", classification_report(Y_test, Y_pred))