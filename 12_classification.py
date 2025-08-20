#Example 2: Spam vs Ham Email Classifier
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Sample dataset
data = {
    "text": [
        "Win money now!!",
        "Hello friend, how are yout?",
        "Free entry in a contest",
        "Are we meeting tomorrow?",
        "Congratulations, you won a prize!"
    ],
    "label": ["spam", "ham", "spam", "ham", "spam"]
}

df = pd.DataFrame(data)

# Convert text to numbers
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(df["text"])
y = df["label"]

# Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train
model = MultinomialNB()
model.fit(x_train, y_train)

# Predict
y_pred = model.predict(x_test)

# Accuracy
print("Accuracy: ",accuracy_score(y_test, y_pred))