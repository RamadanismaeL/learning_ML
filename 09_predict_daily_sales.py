import pandas as pd
import sqlite3  #(change to psycopg2 or mysql.connector for other DBs)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

#1. Connect to Database (SQLite example)
conn = sqlite3.connect("pharmacy.db")

#2. Load data
df = pd.read_sql_query("""
    SELECT date, product_id, qty, price, customers
    FROM sales
""", conn)

print("Data sample:\n", df.head())

#3. Feature Engineering
df["revenue"] = df["qty"] * df["price"] # total revenue
df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek

x = df[["product_id", "qty", "customers", "day_of_week"]]
y = df["revenue"]

#4. Split Data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#5. Train model
model = LinearRegression()
model.fit(x_train, y_train)

#6. Predictions
y_pred = model.predict(x_test)

print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

# 7. Save predictions back to DB
predictions_df = pd.DataFrame({
    "produc_id": x_test["product_id"],
    "actual_revenue": y_test,
    "predicted_revenue": y_pred
})

predictions_df.to_sql("sales_predictions", conn, if_exists="replace", index=False)

print("Predictions saved to 'sales_predictions' table")