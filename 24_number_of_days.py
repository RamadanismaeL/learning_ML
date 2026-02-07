from  sklearn.tree import DecisionTreeClassifier
import numpy as np

X = np.array([
	[50, 2],
	[200, 1],
	[30, 10],
	[150, 3],
	[80, 7],
	[300, 2],
	[40, 14],
	[60, 8],
	[250, 1]
])

y = np.array([1, 1, 0, 1, 0, 1, 0, 1, 0, 1])

model = DecisionTreeClassifier(max_depth=3)
model.fit(X, y)

print("Model trained! Now let's predict for some new customers...\n")

new_customers = np.array([
	[180, 3],
	[45, 12],
	[90, 5],
	[220, 1],
	[25, 20]
])

predictions = model.predict(new_customers)
probabilities = model.predict_proba(new_customers)[:, 1]

print("Results:")
for i in range(len(new_customers)):
	spend, days = new_customers[i]
	prob = probabilities[i]
	decision = "YES - will probabily buy" if predictions[i] == 1 else "NO - probably won't buy"
	print(f" Spend {spend} MT last month, {days} dayas since last top-up -> "
		f"{decision} (confidence {prob:.0%})")