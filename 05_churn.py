# Suponhamos que você trabalha numa operadora de telefonia e quer prever se um cliente vai cancelar o plano (churn)
# Se o modelo prever alta probalidade de cancelamento, o sistema automaticamente cria uma ação para o time de marketing oferecer desconto.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#1. Criar dados fictícios de clientes
data = {
    'meses_como_cliente': [3, 12, 24, 6, 48, 2, 36, 5, 10, 30],
    'valor_mensal': [80, 60, 50, 90, 40, 100, 45, 85, 70, 55],
    'reclamacoes': [2, 0, 1, 3, 0, 4, 0, 2, 1, 0],
    'usou_suport': [1, 0, 0, 1, 0, 1, 0, 1, 0, 0],
    'cancelou': [1, 0, 0, 1, 0, 1, 0, 1, 0, 0] # 1 = cancelou, 0 = ficou
}

df = pd.DataFrame(data)

# 2. Separar features e target
X = df[['meses_como_cliente', 'valor_mensal', 'reclamacoes', 'usou_suporte']]
Y = df['cancelou']

# 3. Treinar modelo
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42)
model = RandomForestClassifier(n_estimators=50, random_state=42)

# 4. Novo cliente chegando
novo_cliente = pd.DataFrame({
    'meses_como_cliente': [4],
    'valor_mensal': [95],
    'reclamacoes': [3],
    'usou_suporte': [1]
})

# 5. Previsão
prob_cancelamento = model.predict_proba(novo_cliente)[0][1] # probabilidade de cancelar

print(f"Probabilidade de cancelamento: {prob_cancelamento:.2%}")

# 6. Decisão automática
if prob_cancelamento > 0.7:
    print("Acionar marketing: oferecer desconto e benefícios extras!")
else:
    print("Cliente em situação estável, sem ação necessária.")
