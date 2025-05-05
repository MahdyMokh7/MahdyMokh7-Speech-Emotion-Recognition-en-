import pandas as pd
import os

# Load the cleaned CSV file
# csv_file_path = r"C:\Users\NoteBook\Desktop\programing\Data Science\Uni project\final project\Features\cleaned_final_features.csv"
FEATURES_FOLDER_PATH = os.path.join(".", "Features")
csv_file_path = os.path.join(FEATURES_FOLDER_PATH, "cleaned_final_features.csv")
df = pd.read_csv(csv_file_path)

# Define the desired column order based on your schema
column_order = [
    'file_name', 'emotion', 'emotion_category',
    
    'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8',
    
    'rms', 'bandwidth', 'zcr',
    
    'spectral_contrast_band1', 'spectral_contrast_band2', 'spectral_contrast_band3', 'spectral_contrast_band4', 
    'spectral_contrast_band5', 'spectral_contrast_band6', 'spectral_contrast_band7',
    
    'chroma1', 'chroma2', 'chroma4', 'chroma5', 'chroma6', 'chroma7', 'chroma8', 'chroma9', 
    'chroma10', 'chroma11', 'chroma12',
    
    # Adding MFCC frame columns from mfcc9 to mfcc13 (assuming the frame columns are present)
    'mfcc9_frame_0', 'mfcc9_frame_1', 'mfcc9_frame_2', 'mfcc9_frame_3', 'mfcc9_frame_4', 'mfcc9_frame_5',
    'mfcc9_frame_6', 'mfcc9_frame_7', 'mfcc9_frame_8', 'mfcc9_frame_9', 'mfcc9_frame_10', 'mfcc9_frame_11',
    'mfcc9_frame_12', 'mfcc9_frame_13', 'mfcc9_frame_14', 'mfcc9_frame_15', 'mfcc9_frame_16', 'mfcc9_frame_17',
    'mfcc9_frame_18', 'mfcc9_frame_19', 'mfcc9_frame_20', 'mfcc9_frame_21', 'mfcc9_frame_22', 'mfcc9_frame_23',
    'mfcc9_frame_24', 'mfcc9_frame_25', 'mfcc9_frame_26', 'mfcc9_frame_27', 'mfcc9_frame_28', 'mfcc9_frame_29',
    'mfcc9_frame_30', 'mfcc9_frame_31', 'mfcc9_frame_32', 'mfcc9_frame_33', 'mfcc9_frame_34', 'mfcc9_frame_35',
    'mfcc9_frame_36', 'mfcc9_frame_37', 'mfcc9_frame_38', 'mfcc9_frame_39', 'mfcc9_frame_40', 'mfcc9_frame_41',
    'mfcc9_frame_42', 'mfcc9_frame_43', 'mfcc9_frame_44', 'mfcc9_frame_45', 'mfcc9_frame_46', 'mfcc9_frame_47',
    'mfcc9_frame_48', 'mfcc9_frame_49', 'mfcc9_frame_50', 'mfcc9_frame_51', 'mfcc9_frame_52', 'mfcc9_frame_53',
    'mfcc9_frame_54', 'mfcc9_frame_55', 'mfcc9_frame_56', 'mfcc9_frame_57', 'mfcc9_frame_58', 'mfcc9_frame_59',
    'mfcc9_frame_60', 'mfcc9_frame_61', 'mfcc9_frame_62',

    # Repeat for mfcc10 to mfcc13
    'mfcc10_frame_0', 'mfcc10_frame_1', 'mfcc10_frame_2', 'mfcc10_frame_3', 'mfcc10_frame_4', 'mfcc10_frame_5',
    'mfcc10_frame_6', 'mfcc10_frame_7', 'mfcc10_frame_8', 'mfcc10_frame_9', 'mfcc10_frame_10', 'mfcc10_frame_11',
    'mfcc10_frame_12', 'mfcc10_frame_13', 'mfcc10_frame_14', 'mfcc10_frame_15', 'mfcc10_frame_16', 'mfcc10_frame_17',
    'mfcc10_frame_18', 'mfcc10_frame_19', 'mfcc10_frame_20', 'mfcc10_frame_21', 'mfcc10_frame_22', 'mfcc10_frame_23',
    'mfcc10_frame_24', 'mfcc10_frame_25', 'mfcc10_frame_26', 'mfcc10_frame_27', 'mfcc10_frame_28', 'mfcc10_frame_29',
    'mfcc10_frame_30', 'mfcc10_frame_31', 'mfcc10_frame_32', 'mfcc10_frame_33', 'mfcc10_frame_34', 'mfcc10_frame_35',
    'mfcc10_frame_36', 'mfcc10_frame_37', 'mfcc10_frame_38', 'mfcc10_frame_39', 'mfcc10_frame_40', 'mfcc10_frame_41',
    'mfcc10_frame_42', 'mfcc10_frame_43', 'mfcc10_frame_44', 'mfcc10_frame_45', 'mfcc10_frame_46', 'mfcc10_frame_47',
    'mfcc10_frame_48', 'mfcc10_frame_49', 'mfcc10_frame_50', 'mfcc10_frame_51', 'mfcc10_frame_52', 'mfcc10_frame_53',
    'mfcc10_frame_54', 'mfcc10_frame_55', 'mfcc10_frame_56', 'mfcc10_frame_57', 'mfcc10_frame_58', 'mfcc10_frame_59',
    'mfcc10_frame_60', 'mfcc10_frame_61', 'mfcc10_frame_62',

    'mfcc11_frame_0', 'mfcc11_frame_1', 'mfcc11_frame_2', 'mfcc11_frame_3', 'mfcc11_frame_4', 'mfcc11_frame_5',
    'mfcc11_frame_6', 'mfcc11_frame_7', 'mfcc11_frame_8', 'mfcc11_frame_9', 'mfcc11_frame_10', 'mfcc11_frame_11',
    'mfcc11_frame_12', 'mfcc11_frame_13', 'mfcc11_frame_14', 'mfcc11_frame_15', 'mfcc11_frame_16', 'mfcc11_frame_17',
    'mfcc11_frame_18', 'mfcc11_frame_19', 'mfcc11_frame_20', 'mfcc11_frame_21', 'mfcc11_frame_22', 'mfcc11_frame_23',
    'mfcc11_frame_24', 'mfcc11_frame_25', 'mfcc11_frame_26', 'mfcc11_frame_27', 'mfcc11_frame_28', 'mfcc11_frame_29',
    'mfcc11_frame_30', 'mfcc11_frame_31', 'mfcc11_frame_32', 'mfcc11_frame_33', 'mfcc11_frame_34', 'mfcc11_frame_35',
    'mfcc11_frame_36', 'mfcc11_frame_37', 'mfcc11_frame_38', 'mfcc11_frame_39', 'mfcc11_frame_40', 'mfcc11_frame_41',
    'mfcc11_frame_42', 'mfcc11_frame_43', 'mfcc11_frame_44', 'mfcc11_frame_45', 'mfcc11_frame_46', 'mfcc11_frame_47',
    'mfcc11_frame_48', 'mfcc11_frame_49', 'mfcc11_frame_50', 'mfcc11_frame_51', 'mfcc11_frame_52', 'mfcc11_frame_53',
    'mfcc11_frame_54', 'mfcc11_frame_55', 'mfcc11_frame_56', 'mfcc11_frame_57', 'mfcc11_frame_58', 'mfcc11_frame_59',
    'mfcc11_frame_60', 'mfcc11_frame_61', 'mfcc11_frame_62',

    'mfcc12_frame_0', 'mfcc12_frame_1', 'mfcc12_frame_2', 'mfcc12_frame_3', 'mfcc12_frame_4', 'mfcc12_frame_5',
    'mfcc12_frame_6', 'mfcc12_frame_7', 'mfcc12_frame_8', 'mfcc12_frame_9', 'mfcc12_frame_10', 'mfcc12_frame_11',
    'mfcc12_frame_12', 'mfcc12_frame_13', 'mfcc12_frame_14', 'mfcc12_frame_15', 'mfcc12_frame_16', 'mfcc12_frame_17',
    'mfcc12_frame_18', 'mfcc12_frame_19', 'mfcc12_frame_20', 'mfcc12_frame_21', 'mfcc12_frame_22', 'mfcc12_frame_23',
    'mfcc12_frame_24', 'mfcc12_frame_25', 'mfcc12_frame_26', 'mfcc12_frame_27', 'mfcc12_frame_28', 'mfcc12_frame_29',
    'mfcc12_frame_30', 'mfcc12_frame_31', 'mfcc12_frame_32', 'mfcc12_frame_33', 'mfcc12_frame_34', 'mfcc12_frame_35',
    'mfcc12_frame_36', 'mfcc12_frame_37', 'mfcc12_frame_38', 'mfcc12_frame_39', 'mfcc12_frame_40', 'mfcc12_frame_41',
    'mfcc12_frame_42', 'mfcc12_frame_43', 'mfcc12_frame_44', 'mfcc12_frame_45', 'mfcc12_frame_46', 'mfcc12_frame_47',
    'mfcc12_frame_48', 'mfcc12_frame_49', 'mfcc12_frame_50', 'mfcc12_frame_51', 'mfcc12_frame_52', 'mfcc12_frame_53',
    'mfcc12_frame_54', 'mfcc12_frame_55', 'mfcc12_frame_56', 'mfcc12_frame_57', 'mfcc12_frame_58', 'mfcc12_frame_59',
    'mfcc12_frame_60', 'mfcc12_frame_61', 'mfcc12_frame_62',

    'mfcc13_frame_0', 'mfcc13_frame_1', 'mfcc13_frame_2', 'mfcc13_frame_3', 'mfcc13_frame_4', 'mfcc13_frame_5',
    'mfcc13_frame_6', 'mfcc13_frame_7', 'mfcc13_frame_8', 'mfcc13_frame_9', 'mfcc13_frame_10', 'mfcc13_frame_11',
    'mfcc13_frame_12', 'mfcc13_frame_13', 'mfcc13_frame_14', 'mfcc13_frame_15', 'mfcc13_frame_16', 'mfcc13_frame_17',
    'mfcc13_frame_18', 'mfcc13_frame_19', 'mfcc13_frame_20', 'mfcc13_frame_21', 'mfcc13_frame_22', 'mfcc13_frame_23',
    'mfcc13_frame_24', 'mfcc13_frame_25', 'mfcc13_frame_26', 'mfcc13_frame_27', 'mfcc13_frame_28', 'mfcc13_frame_29',
    'mfcc13_frame_30', 'mfcc13_frame_31', 'mfcc13_frame_32', 'mfcc13_frame_33', 'mfcc13_frame_34', 'mfcc13_frame_35',
    'mfcc13_frame_36', 'mfcc13_frame_37', 'mfcc13_frame_38', 'mfcc13_frame_39', 'mfcc13_frame_40', 'mfcc13_frame_41',
    'mfcc13_frame_42', 'mfcc13_frame_43', 'mfcc13_frame_44', 'mfcc13_frame_45', 'mfcc13_frame_46', 'mfcc13_frame_47',
    'mfcc13_frame_48', 'mfcc13_frame_49', 'mfcc13_frame_50', 'mfcc13_frame_51', 'mfcc13_frame_52', 'mfcc13_frame_53',
    'mfcc13_frame_54', 'mfcc13_frame_55', 'mfcc13_frame_56', 'mfcc13_frame_57', 'mfcc13_frame_58', 'mfcc13_frame_59',
    'mfcc13_frame_60', 'mfcc13_frame_61', 'mfcc13_frame_62',

    'centroid_frame_0', 'centroid_frame_1', 'centroid_frame_2', 'centroid_frame_3', 'centroid_frame_4', 'centroid_frame_5',
    'centroid_frame_6', 'centroid_frame_7', 'centroid_frame_8', 'centroid_frame_9', 'centroid_frame_10', 'centroid_frame_11',
    'centroid_frame_12', 'centroid_frame_13', 'centroid_frame_14', 'centroid_frame_15', 'centroid_frame_16', 'centroid_frame_17',
    'centroid_frame_18', 'centroid_frame_19', 'centroid_frame_20', 'centroid_frame_21', 'centroid_frame_22', 'centroid_frame_23',
    'centroid_frame_24', 'centroid_frame_25', 'centroid_frame_26', 'centroid_frame_27', 'centroid_frame_28', 'centroid_frame_29',
    'centroid_frame_30', 'centroid_frame_31', 'centroid_frame_32', 'centroid_frame_33', 'centroid_frame_34', 'centroid_frame_35',
    'centroid_frame_36', 'centroid_frame_37', 'centroid_frame_38', 'centroid_frame_39', 'centroid_frame_40', 'centroid_frame_41',
    'centroid_frame_42', 'centroid_frame_43', 'centroid_frame_44', 'centroid_frame_45', 'centroid_frame_46', 'centroid_frame_47',
    'centroid_frame_48', 'centroid_frame_49', 'centroid_frame_50', 'centroid_frame_51', 'centroid_frame_52', 'centroid_frame_53',
    'centroid_frame_54', 'centroid_frame_55', 'centroid_frame_56', 'centroid_frame_57', 'centroid_frame_58', 'centroid_frame_59',
    'centroid_frame_60', 'centroid_frame_61', 'centroid_frame_62',

    'rolloff_frame_0', 'rolloff_frame_1', 'rolloff_frame_2', 'rolloff_frame_3', 'rolloff_frame_4', 'rolloff_frame_5',
    'rolloff_frame_6', 'rolloff_frame_7', 'rolloff_frame_8', 'rolloff_frame_9', 'rolloff_frame_10', 'rolloff_frame_11',
    'rolloff_frame_12', 'rolloff_frame_13', 'rolloff_frame_14', 'rolloff_frame_15', 'rolloff_frame_16', 'rolloff_frame_17',
    'rolloff_frame_18', 'rolloff_frame_19', 'rolloff_frame_20', 'rolloff_frame_21', 'rolloff_frame_22', 'rolloff_frame_23',
    'rolloff_frame_24', 'rolloff_frame_25', 'rolloff_frame_26', 'rolloff_frame_27', 'rolloff_frame_28', 'rolloff_frame_29',
    'rolloff_frame_30', 'rolloff_frame_31', 'rolloff_frame_32', 'rolloff_frame_33', 'rolloff_frame_34', 'rolloff_frame_35',
    'rolloff_frame_36', 'rolloff_frame_37', 'rolloff_frame_38', 'rolloff_frame_39', 'rolloff_frame_40', 'rolloff_frame_41',
    'rolloff_frame_42', 'rolloff_frame_43', 'rolloff_frame_44', 'rolloff_frame_45', 'rolloff_frame_46', 'rolloff_frame_47',
    'rolloff_frame_48', 'rolloff_frame_49', 'rolloff_frame_50', 'rolloff_frame_51', 'rolloff_frame_52', 'rolloff_frame_53',
    'rolloff_frame_54', 'rolloff_frame_55', 'rolloff_frame_56', 'rolloff_frame_57', 'rolloff_frame_58', 'rolloff_frame_59',
    'rolloff_frame_60', 'rolloff_frame_61', 'rolloff_frame_62',

    'chroma3_frame_0', 'chroma3_frame_1', 'chroma3_frame_2', 'chroma3_frame_3', 'chroma3_frame_4', 'chroma3_frame_5',
    'chroma3_frame_6', 'chroma3_frame_7', 'chroma3_frame_8', 'chroma3_frame_9', 'chroma3_frame_10', 'chroma3_frame_11',
    'chroma3_frame_12', 'chroma3_frame_13', 'chroma3_frame_14', 'chroma3_frame_15', 'chroma3_frame_16', 'chroma3_frame_17',
    'chroma3_frame_18', 'chroma3_frame_19', 'chroma3_frame_20', 'chroma3_frame_21', 'chroma3_frame_22', 'chroma3_frame_23',
    'chroma3_frame_24', 'chroma3_frame_25', 'chroma3_frame_26', 'chroma3_frame_27', 'chroma3_frame_28', 'chroma3_frame_29',
    'chroma3_frame_30', 'chroma3_frame_31', 'chroma3_frame_32', 'chroma3_frame_33', 'chroma3_frame_34', 'chroma3_frame_35',
    'chroma3_frame_36', 'chroma3_frame_37', 'chroma3_frame_38', 'chroma3_frame_39', 'chroma3_frame_40', 'chroma3_frame_41',
    'chroma3_frame_42', 'chroma3_frame_43', 'chroma3_frame_44', 'chroma3_frame_45', 'chroma3_frame_46', 'chroma3_frame_47',
    'chroma3_frame_48', 'chroma3_frame_49', 'chroma3_frame_50', 'chroma3_frame_51', 'chroma3_frame_52', 'chroma3_frame_53',
    'chroma3_frame_54', 'chroma3_frame_55', 'chroma3_frame_56', 'chroma3_frame_57', 'chroma3_frame_58', 'chroma3_frame_59',
    'chroma3_frame_60', 'chroma3_frame_61', 'chroma3_frame_62'

]

print("started reordering..")
# Reorder columns
df = df[column_order]

# Save the reordered DataFrame to a new CSV
# cleaned_and_ordered_csv_file_path = r"C:\Users\NoteBook\Desktop\programing\Data Science\Uni project\final project\Features\final_features_ImportReady.csv"
cleaned_and_ordered_csv_file_path = os.path.join(FEATURES_FOLDER_PATH, "final_features_ImportReady.csv")
df.to_csv(cleaned_and_ordered_csv_file_path, index=False)

print(f"Reordered CSV saved at: {cleaned_and_ordered_csv_file_path}")
