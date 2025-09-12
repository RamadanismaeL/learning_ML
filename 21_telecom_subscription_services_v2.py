import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Criar um dataset fictício, mas realista
data = pd.DataFrame({
    "Age": [25, 40, 35, 50, 23, 44, 36, 29, 60, 33],
    "MonthlySpend": [50, 120, 80, 200, 30, 150, 90, 60, 220, 85],
    "YearsWithCompany": [1, 10, 5, 20, 0.5, 12, 6, 2, 25, 4],
    "SupportTickets": [0, 2, 1, 5, 0, 3, 1, 0, 6, 1],
    "Churn": [0, 0, 0, 1, 0, 0, 0, 0, 1, 0] # 1 = cliente cancelou, o = manteve
})

# Separar features (X) e target (y)
X = data.drop("Churn", axis=1)
y = data["Churn"]

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Escalar os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.values)
X_test_scaled = scaler.transform(X_test.values)

# Treinar modelo Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Fazer previsões
y_pred = model.predict(X_test_scaled)

# Acaliar modelo
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Acurácia:", accuracy)
print("Matriz de Confusão:\n", cm)
print("Relatório de Classificação:\n", report)

# Prever churn de um novo cliente
novo_cliente = [[30, 75, 3, 1]]
novo_cliente_scaled = scaler.transform(novo_cliente)
previsao = model.predict(novo_cliente_scaled)
print("O cliente vai cancelar? " , "Sim" if previsao[0] == 1 else "Não")