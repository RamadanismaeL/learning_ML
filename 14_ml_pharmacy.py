import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

#1. Criar dados fictícios
data = {
    'produto_id': [1,2,3,4,5],
    'vendas_ult_7_dias': [10, 2, 15, 5, 0],
    'vendas_ult_30_dias': [50, 10, 60, 20, 5],
    'estoque_atual': [5, 8, 2, 10, 1],
    'dias_para_falta': [1, 5, 0, 10, 0]  # target
}

df = pd.DataFrame(data)

df['vai_faltar'] = np.where(df['dias_para_falta'] < 3, 1, 0)
df.drop('dias_para_falta', axis=1, inplace=True)

print("Dados iniciais:")
print(df)

#2. Separa features e target
x = df[['vendas_ult_7_dias', 'vendas_ult_30_dias', 'estoque_atual']]
y = df['vai_faltar']

#3. Dividir treino e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

#4. Treinar modelo Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

#5. Fazer previsões no teste
y_pred = model.predict(x_test)

#6. Avaliar modelo
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#7. Previsão para novo produto
novo_produto = pd.DataFrame({
    'vendas_ult_7_dias': [7],
    'vendas_ult_30_dias': [30],
    'estoque_atual': [3]
})

previsao = model.predict(novo_produto)
print("\n🔮 Previsão para novo produto:")
print("Vai faltar?" , "Sim" if previsao[0]==1 else "Não")