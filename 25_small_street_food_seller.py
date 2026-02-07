# =============================================================================
#   NEW SIMPLE ML EXAMPLE – Street food seller edition
#   Predict: "Will I sell out of chamussas today? Yes / No"
# =============================================================================

from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Fake data from past days (you could collect real data in a notebook!)
# Features: [day_of_week (1=Mon ... 7=Sun), temp_C]  → sold_out (1=Yes, 0=No)
X = np.array([
    [1, 28],   # Monday, mild     → usually no rush
    [2, 30],
    [3, 32],   # mid-week, hot    → good sales
    [4, 34],
    [5, 33],   # Friday, hot      → often sells out
    [6, 31],   # Saturday         → busy
    [7, 29],   # Sunday           → medium
    [5, 36],   # very hot Friday  → sold out fast
    [6, 35],
    [1, 27],
    [3, 31],
    [7, 34],
    [4, 29],   # Thursday, cooler → didn't sell out
    [2, 28],
])

y = np.array([0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0])

# Train simple model
model = DecisionTreeClassifier(max_depth=3)
model.fit(X, y)

print("Model trained on past sales!\nNow checking today's forecast...\n")

# Today's situation (change these two numbers to test different days/weather)
today = np.array([
    [5, 35],   # Friday + very hot
    [1, 29],   # Monday + mild
    [6, 33],   # Saturday + warm
    [3, 30],   # Wednesday + normal
    [7, 36],   # Sunday + very hot
    [4, 28],   # Thursday + cooler
])

predictions = model.predict(today)
probabilities = model.predict_proba(today)[:, 1]   # probability of selling out (Yes)

print("Today's sell-out predictions:")
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

for i in range(len(today)):
    dow, temp = today[i]
    day_name = days[int(dow)-1]
    prob = probabilities[i]
    result = "YES – probably sell out! 🔥" if predictions[i] == 1 else "NO – likely some left"
    print(f"  {day_name}, {temp}°C → {result}  (probability {prob:.0%})")