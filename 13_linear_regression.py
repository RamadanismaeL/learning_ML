import sqlite3
import random
import datetime
import pandas as pd
from sklearn.linear_model import LinearRegression


#=========================
# STEP 1: CREATE DATABASE
#=========================
# Create SQLite DB
conn = sqlite3.connect("pharmacy2.db")
cursor = conn.cursor()

# Create tables
cursor.execute("""
CREATE TABLE IF NOT EXISTS sales (
    id INTEGER PRIMARY KEY,
    product TEXT,
    date TEXT,
    quantity INTEGER,
    price REAL
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS inventory (
    product TEXT PRIMARY KEY,
    stock INTEGER,
    reorder_level INTEGER
)
""")

# Insert sample inventory
inventory_data = [
    ("Paracetamol", 200, 50),
    ("Amoxicillin", 120, 30),
    ("Vitamin C", 300, 100),
    ("Ibuprofen", 150, 40)
]

cursor.executemany("INSERT OR REPLACE INTO inventory VALUES (?, ?, ?)", inventory_data)

# Generate sales data (last 90 days)
products = ["Paracetamol", "Amoxicillin", "Vitamin C", "Ibuprofen"]
start_date = datetime.date.today() - datetime.timedelta(days=90)

sales_data = []
for day in range(90):
    date = start_date + datetime.timedelta(days=day)
    for product in products:
        quantity = random.randint(0, 20)  # sold per day
        price = random.uniform(2.0, 10.0)
        sales_data.append((product, date.isoformat(), quantity, price))

cursor.executemany("INSERT INTO sales (product, date, quantity, price) VALUES (?, ?, ?, ?)", sales_data)

conn.commit()
print("Database created with sales + inventory!")


# STEP 2: LOAD DATA
df = pd.read_sql_query("SELECT * FROM sales", conn)

# Aggregate daily sales per product
df_grouped = df.groupby(["date", "product"])["quantity"].sum().reset_index()
df_grouped["date"] = pd.to_datetime(df_grouped["date"])
df_grouped["day_num"] = (df_grouped["date"] - df_grouped["date"].min()).dt.days

# STEP 3: TRAIN FORECAST MODEL
models = {}
for product in df_grouped["product"].unique():
    product_data = df_grouped[df_grouped["product"] == product]
    x = product_data[["day_num"]]
    y = product_data["quantity"]

    model = LinearRegression()
    model.fit(x, y)

    models[product] = model

print("Forecast models trained for each medicine\n")

# STEP 4: PREDICT STOCK OUT
def predict_stock_out(product, current_stock):
    model = models[product]

    future_days = [[df_grouped["day_num"].max() + i] for i in range (1, 31)]
    predictions = model.predict(future_days)

    total_expected_sales = sum(predictions)

    if total_expected_sales >= current_stock:
        return f"{product} will run out in less than 30 days!"
    else:
        return f"{product} stock is sufficient for next 30 days."
    

# STEP 5: reorder suggestion
def reorder_suggestion(product, stock, reorder_level):
    if stock <= reorder_level:
        return f"Reorder needed for {product}! Suggest quantity: {reorder_level*2}"
    else:
        return f"{product} has enough stock."
    

# STEP 6: RUN REPORT
inventory_df = pd.read_sql_query("SELECT * FROM inventory", conn)

print("STOCK PREDICTION RESULTS:")
for _, row in inventory_df.iterrows():
    print(predict_stock_out(row["product"], row["stock"]))

print("\nREORDER SUGGESTIONS:")
for _, row in inventory_df.iterrows():
    print(reorder_suggestion(row["product"], row["stock"], row["reorder_level"]))