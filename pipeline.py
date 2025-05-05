import subprocess 
import os

# Pre-process
subprocess.run(["python", r"C:\Users\Mahmodiyan-PC\Desktop\agha alborz\term6\datascience\final project\Speech-Emotion-Recognition-en-\scripts\preprocess.py"])

# Feature Extraction
subprocess.run(["python", r"C:\Users\Mahmodiyan-PC\Desktop\agha alborz\term6\datascience\final project\Speech-Emotion-Recognition-en-\scripts\feature_extraction.py"])

# Import features to DB
# arg_file_path = r"C:\Users\NoteBook\Desktop\programing\Data Science\Uni project\final project\Features\final_features_ImportReady.csv"
arg_file_path = os.path.join(".", "Features", "final_features_ImportReady.csv")
subprocess.run(["python", "./scripts/import_to_db.py", arg_file_path])

# Load Data to DF
# subprocess.run(["python", "./scripts/load_data.py"])

# # Feature Engineering
# subprocess.run(["python", "./scripts/feature_engineering.py"])

# # Import Engineered featuers to DB
# # arg_eng_file_path = r"C:\Users\NoteBook\Desktop\programing\Data Science\Uni project\final project\Features\final_features_Finalized.csv
# arg_eng_file_path = os.path.join(".", "Features", "final_features_Finalized.csv")
# subprocess.run(["python", "./scripts/import_to_db.py", arg_eng_file_path])

