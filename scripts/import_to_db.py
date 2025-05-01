import sys
import pandas as pd
from database_connection import get_connection

if len(sys.argv) > 1:
    csv_file_path = sys.argv[1] 
    print(f"CSV file path received: {csv_file_path}")
else:
    print("No CSV file path argument provided.")
    sys.exit(1)  # Exit if no argument is provided

df = pd.read_csv(csv_file_path)

engine = get_connection()

df.head(10).to_sql(name='features', con=engine, if_exists='replace', index=False)

print("DataFrame written to MySQL table")
