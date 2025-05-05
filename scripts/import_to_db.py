import sys
import pandas as pd
from database_connection import get_connection

print("import_to_db.py argumnet recieved:  ", sys.argv[1])

if len(sys.argv) > 1:
    csv_file_path = sys.argv[1] 
    print(csv_file_path)
    print(f"CSV file path received: {csv_file_path}")
else:
    print("No CSV file path argument provided.")
    sys.exit(1)  # Exit if no argument is provided

df = pd.read_csv(csv_file_path)

engine = get_connection()
if engine is None:
    print("Database connection failed. Exiting.")
    exit(1)


print(f"moving the csv file {sys.argv[1]}  to the databse")

df.to_sql(name='features', con=engine, if_exists='replace', index=False)

print("DataFrame written to MySQL table")
