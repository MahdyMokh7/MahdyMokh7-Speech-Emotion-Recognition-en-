import pandas as pd
import numpy as np
import os

# Load the CSV file
csv_file_path = r"C:\Users\NoteBook\Desktop\programing\Data Science\Uni project\final project\Features\final_features.csv"
print(f"CSV file path: {csv_file_path}")

# Check if the file exists
if os.path.exists(csv_file_path):
    print("File found.")
    df = pd.read_csv(csv_file_path)  # Load the data into the DataFrame
else:
    print("File not found.")
    # Stop the execution if the file is not found
    exit()

# Define the emotion mapping
emotion_mapping = {
    0: 'ANG',
    1: 'DIS',
    2: 'FEA',
    3: 'HAP',
    4: 'NEU',
    5: 'SAD'
}

# Function to process multi-value columns
def process_multi_value_column(col):
    """
    This function processes columns that contain multi-value entries like 'np.float64(...)' 
    and returns them as a list of float values.
    """
    # Convert strings like 'np.float64(...)' to actual float values
    return [float(x.split('=')[1].replace(")", "").strip()) for x in col]

# Process the multi-value columns and create new columns for each frame
for column in df.columns:
    if df[column].dtype == 'object':  # Likely a multi-value column
        # Check if the column contains multi-value data
        if "np.float64" in str(df[column].iloc[0]):
            # Process the multi-value column (e.g., MFCCs, ZCR frames, etc.)
            processed_values = df[column].apply(process_multi_value_column)
            
            # Create individual frame columns for each frame in the multi-value list
            for i in range(63):  # Assuming there are 63 frames
                df[f"{column}_frame_{i}"] = processed_values.apply(lambda x: x[i] if len(x) > i else None)

# Create emotion_category based on the 'emotion' column
df['emotion_category'] = df['emotion'].map(emotion_mapping)

# Clean the CSV to match the table schema (remove original multi-value columns)
columns_to_remove = [col for col in df.columns if 'np.float64' in str(df[col].iloc[0])]
df_cleaned = df.drop(columns=columns_to_remove)

# Save the cleaned DataFrame to a new CSV
cleaned_csv_file_path = r"C:\Users\NoteBook\Desktop\programing\Data Science\Uni project\final project\Features\cleaned_final_features.csv"
df_cleaned.to_csv(cleaned_csv_file_path, index=False)

print(f"Cleaned CSV saved at: {cleaned_csv_file_path}")
