# =============================================================================
#   Super simple ML example – runs in one code block
#   Predicts: "Will buy a new phone? Yes / No"
# =============================================================================

# 1. Import only what we need (these come with most Python environments)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# 2. Tiny fake dataset (10 people) – you can add more rows later!
# Columns: [age, income_MZN, already_has_smartphone (1=Yes, 0=No)]  →  will_buy (1=Yes, 0=No)
data = np.array([
    [22,  15000,  0,  1],   # young, low income, no phone → buys
    [19,   8000,  0,  1],
    [35,  45000,  1,  0],   # older, good income, already has phone → no
    [28,  22000,  1,  0],
    [24,  18000,  0,  1],
    [41,  60000,  1,  0],
    [20,  12000,  0,  1],
    [33,  30000,  1,  1],   # has phone but high income → still buys
    [26,  25000,  0,  1],
    [50,  80000,  1,  0],
])

# Split into features (X) and target (y)
X = data[:, :3]           # first 3 columns
y = data[:, 3]            # last column = will_buy

# 3. Split into train + test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Create and train a very simple decision tree
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# 5. Check accuracy on the small test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set: {accuracy:.0%}  (small data = not very reliable yet)")

# 6. Predict for new people (real use case!)
new_customers = np.array([
    [23,  14000,  0],     # young Maputo student, no phone
    [38,  55000,  1],     # office worker, already has good phone
    [21,   9000,  0],     # another student
    [29,  32000,  1],     # mid income, has phone
])

predictions = model.predict(new_customers)
probabilities = model.predict_proba(new_customers)[:, 1]   # probability of Yes

print("\nPredictions for new customers:")
for i, (age, income, has_phone) in enumerate(new_customers):
    yes_prob = probabilities[i]
    decision = "YES ✅" if predictions[i] == 1 else "NO ❌"
    print(f"Age {age}, income {income:,} MZN, has_phone={bool(has_phone)} → "
          f"{decision}  (prob {yes_prob:.0%})")