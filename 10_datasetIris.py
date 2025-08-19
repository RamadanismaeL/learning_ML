from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

#1. Carregar o dataset Iris
iris = load_iris()
x = iris.data # Características (comprimento pétala, largura pétala, etc)
y = iris.target # Classes (0,1,2 -> tipos de flores)

#2. Dividir em treino e teste (80% treino, 20% teste)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#3. Criar o modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)

#4. Treinar o modelo
model.fit(x_train, y_train)

#5. Fazer previsões
y_pred = model.predict(x_test)

#6. Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

#7. Fazer previsão de um novo exemplo
new_sample = [[5.1, 3.5, 1.4, 0.2]] # características de uma nova flor
predicted_class = iris.target_names[model.predict(new_sample)[0]]
print(f"\nNova previsão: {predicted_class}")

# Explicação passo a passo
#1. load_iris() -> carrega dados de flores (4 características, 3 classes)
#2. train_test_split -> divide em conjunto de treino e teste
#3. RandomForestClassifier -> modelo de classificação robusto
#4. model.fit() -> treina o modelo com os dados de treino
#5. model.predict() -> faz previsões em novos dados
#6. accuracy_score / classification_report -> avalia a perfomance do modelo
#7. Nova previsão -> mostra como usar o modelo para prever uma nova entrada