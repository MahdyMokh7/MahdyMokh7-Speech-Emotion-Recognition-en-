import pandas as pd
from database_connection import get_connection
from sklearn.preprocessing import StandardScaler

def load_data_from_db() -> pd.DataFrame:
    engine = get_connection()
    query = "SELECT * FROM features"
    df = pd.read_sql(query, con=engine)
    return df



df_final_features = load_data_from_db()

meta_cols = ['file_name', 'emotion', 'emotion_category']
feature_cols = df_final_features.columns.difference(meta_cols, sort=False)

# Apply StandardScaler only on feature columns
scaler = StandardScaler()
df_final_features[feature_cols] = scaler.fit_transform(df_final_features[feature_cols])

df_final_features.to_csv(r"F:\csv_features\final_engineered_features.csv")

print("final features have been succesfully engineered")

    

