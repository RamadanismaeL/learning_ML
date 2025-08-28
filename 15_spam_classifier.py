# Simple Spam vs Ham Email Classifier using Logistic Regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# load dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep="\t", header=None, names=["label", "message"])

print("Dataset loaded successfully.")
print(df.head(), "\n\n")

# preprocess data
x = df["message"]
y = df["label"]

# split dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# convert text to numeric vectors
vectorizer = CountVectorizer()
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

# train model
model = LogisticRegression(max_iter=1000)
model.fit(x_train_vec, y_train)

print("Model trained successfully.\n")

# evaluate model
y_pred = model.predict(x_test_vec)

print("Evaluation Metrics:")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, pos_label="spam"))
print("Recall   :", recall_score(y_test, y_pred, pos_label="spam"))
print("F1 Score :", f1_score(y_test, y_pred, pos_label="spam"))

# confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])
print("\nConfusion Matrix:\n", cm)

# visualize confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["ham", "spam"],
    yticklabels=["ham", "spam"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()