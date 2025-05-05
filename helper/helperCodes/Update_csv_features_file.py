import pandas as pd
import ast
import numpy as np
import os

# Load the CSV file
# csv_file_path = r"C:\Users\NoteBook\Desktop\programing\Data Science\Uni project\final project\Features\final_features.csv"
FEATURES_FOLDER_PATH = os.path.join(".", "Features")
csv_file_path = os.path.join(FEATURES_FOLDER_PATH ,"final_features.csv")
df = pd.read_csv(csv_file_path)

print("csv read completed.")

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
    and returns them as a list of regular float values.
    """
    processed_values = []
    for x in col:
        try:
            # Remove 'np.float64(' and ')' and convert the value to float
            # This will clean the np.float64() format
            values = [float(val.split('(')[-1].split(')')[0]) for val in x.strip('[]').split(',')] 
            processed_values.append(values)  # Convert to list of regular floats
        except Exception as e:
            print("Exception occurred, assigning None values.")
            processed_values.append([None] * 63)  # Fill with None if error occurs
    return processed_values


# List of multi-value columns based on the provided update
multi_value_columns = ['mfcc9', 'mfcc10', 'mfcc11', 'mfcc12', 'mfcc13', 'centroid', 'rolloff', 'chroma3']

# Create an empty list to collect all new frame columns
frame_columns = []

print("started processing the multi value columns...")
# Process the multi-value columns
for column in multi_value_columns:
    processed_values = process_multi_value_column(df[column])
    
    # Create 63 new columns for each frame and add them to the frame_columns list
    for i in range(63):  # Assuming there are 63 frames
        frame_columns.append(pd.Series([x[i] if len(x) > i else None for x in processed_values], name=f"{column}_frame_{i}"))
    
    # Drop the original multi-value column after processing
    df.drop(column, axis=1, inplace=True)

# Concatenate all the frame columns to the original DataFrame
df = pd.concat([df] + frame_columns, axis=1)

# Create emotion_category based on the 'emotion' column
df['emotion_category'] = df['emotion'].map(emotion_mapping)


print("started writing in the csv...")
# Save the cleaned DataFrame to a new CSV
# cleaned_csv_file_path = r"C:\Users\NoteBook\Desktop\programing\Data Science\Uni project\final project\Features\cleaned_final_features.csv"
cleaned_csv_file_path = os.path.join(FEATURES_FOLDER_PATH, "cleaned_final_features.csv")
df.to_csv(cleaned_csv_file_path, index=False)

print(f"Cleaned CSV saved at: {cleaned_csv_file_path}")
