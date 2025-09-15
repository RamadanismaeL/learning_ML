# Detectando fraudes em cartão de crédito
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Carregar dataset público de fraudes
# Dataset: https://www.kaggle.com/mlg-ulb/creditcardfraud
# Obs.: contém 284,807 transações, das quais apenas 492 são fraudes (~0.17%)
df = pd.read_csv("creditcard.csv")

print("Formato do dataset:", df.shape)
print("Fraudes:", df['Class'].sum())

# 2. Separar freatures (X) e target (y)
X = df.drop("Class", axis=1)
y = df["Class"]

# 3. Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# 4. Criar modelo (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Fazer previsões
y_pred = model.predict(X_test)

# 6. Avaliar
print("\n Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))