import pandas as pd
from database_connection import get_connection

def load_data_from_db() -> pd.DataFrame:
    engine = get_connection()
    query = "SELECT * FROM features"
    df = pd.read_sql(query, con=engine)
    return df



if __name__ == "__main__":
    df_final_features = load_data_from_db()
    

