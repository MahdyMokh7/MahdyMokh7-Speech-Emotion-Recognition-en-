import pandas as pd
from database_connection import get_connection

engine = get_connection()

df = pd.read_sql("SELECT * FROM features", con=engine)

print("Database Loaded successfuly.")
print(df.head(5))