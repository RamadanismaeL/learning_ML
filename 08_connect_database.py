import sqlite3
import pandas as pd
import random
from datetime import datetime, timedelta

#1. Create and connect to SQLite DB
conn = sqlite3.connect("pharmacy.db")
cursor = conn.cursor()

#2. Create sales table
cursor.execute("""
CREATE TABLE IF NOT EXISTS sales (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT,
    product_id INTEGER,
    qty INTEGER,
    price REAL,
    customers INTEGER
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS sales_predictions (
    product_id INTEGER,
    actual_revenue REAL,
    predicted_revenue REAL
)
""")

#3. Insert sample data (simulate 180 days sales for 5 products)
products = {
    1: 10.0, # Paracetamol
    2: 25.0, # Vitamin C
    3: 50.0, # Antibiotic
    4: 5.0,  # Aspirin
    5: 15.0  # Cough Syrup
}

start_date = datetime.today() - timedelta(days=180)
data = []

for day in range(180):
    date = (start_date + timedelta(days=day)).strftime("%Y-%m-%d")
    customers = random.randint(20, 100)
    for product_id, price in products.items():
        qty = random.randint(1, 10) * customers // 10
        data.append((date, product_id, qty, price, customers))
    

cursor.executemany("INSERT INTO sales (date, product_id, qty, price, customers) VALUES (?, ?, ?, ?, ?)", data)

# 4. Commit and close
conn.commit()

#5. Check sample
df = pd.read_sql_query("SELECT * FROM sales LIMIT 10", conn)
print(df)

conn.close()