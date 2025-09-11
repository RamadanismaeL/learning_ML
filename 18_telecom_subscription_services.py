import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Load and prepare the dataset
# Sample dataset (in practice, you'd load from a CSV or database)
data = {
    'tenure': [1, 34, 2, 45, 8, 22, 10, 28, 14, 60],
    'monthly_charges': [29.85, 59.95, 53.85, 42.30, 70.70, 99.65, 89.10, 29.75, 104.80, 56.15],
    'contract_length': [1, 12, 1, 24, 1, 12, 1, 12, 1, 24],
    'churn': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
}
df = pd.DataFrame(data)
#print(df)

# 2. Data preprocessing
# Features and target
X = df.drop('churn', axis=1) # axis = 0 = rows, axis=1=columns
y = df['churn']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scake numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.values)
X_test_scaled = scaler.transform(X_test.values)

# 3. Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 4. Make predictions
y_pred = model.predict(X_test_scaled)

# 5. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy: .2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# 6. Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance')
plt.show()

# 7. Example prediction for a new customer
new_customer = np.array([[15, 75.50, 12]]) # tensure, monthly_charges, contract_length
new_customer_scaled = scaler.transform(new_customer)
prediction = model.predict(new_customer_scaled)
print(f"\nPrediction for new customer: {'Churn' if prediction [0] == 1 else 'No Churn'}")