import os
import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
import seaborn as sns
import ast
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, f_classif
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns




# === CONFIGURATION ===
preprocessed_dataset_path = os.path.join("..", "PreProcessedDataSet_for_ML")
FEATURE_FOLDER_PATH = os.path.join("..", "Features")
csv_features_path = os.path.join(FEATURE_FOLDER_PATH, "audio_features.csv")
csv_features_path_separated = os.path.join(FEATURE_FOLDER_PATH, "audio_features_separated.csv")
csv_aggregated_features_path = os.path.join(FEATURE_FOLDER_PATH, "audio_aggregated_features.csv")
csv_final_features_path = os.path.join(FEATURE_FOLDER_PATH, "final_features.csv")
N_FFT = 512 # 32 ms window 
HOP_LENGTH = 205 # 12.8 ms step → 60% overlap
SR=16000 
N_MFCC=13 
os.makedirs(FEATURE_FOLDER_PATH, exist_ok=True)





# =================== FEATURE EXCTRACTION FUNCTIONS WE USED ===================
def compute_mfcc(y, sr, n_mfcc, n_fft, hop_length):

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return mfcc

def compute_zcr(y, n_fft, hop_length):
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)
    return zcr

def compute_rms(y, n_fft, hop_length):
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)
    return rms

def compute_centroid(y, sr, n_fft, hop_length):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    return centroid

def compute_bandwidth(y, sr, n_fft, hop_length):
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    return bandwidth

def compute_rolloff(y, sr, n_fft, hop_length):
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    return rolloff

def compute_spectral_contrast(y, sr, n_fft, hop_length):
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    return spectral_contrast

def compute_chroma(y, sr, n_fft, hop_length):
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    return chroma

def compute_spectral_flux(y, sr, n_fft, hop_length):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0))
    return flux






def extract_features_from_audio(file_path):
    try:
        y, _ = librosa.load(file_path, sr=SR)
        
        # MFCCs, MFCC Delta & Delta-Delta
        mfcc = compute_mfcc(y, SR, N_MFCC, N_FFT, HOP_LENGTH)

        # Zero Crossing Rate
        zcr = compute_zcr(y, N_FFT, HOP_LENGTH)

        # RMS Energy
        rms = compute_rms(y, N_FFT, HOP_LENGTH)

        # Spectral Features (only mean)
        centroid = compute_centroid(y, SR, N_FFT, HOP_LENGTH)
        bandwidth = compute_bandwidth(y, SR, N_FFT, HOP_LENGTH)
        rolloff = compute_rolloff(y, SR, N_FFT, HOP_LENGTH)

        # Spectral Contrast (mean + std)
        contrast = compute_spectral_contrast(y, SR, N_FFT, HOP_LENGTH)

        # Chroma Features (mean only)
        chroma = compute_chroma(y, SR, N_FFT, HOP_LENGTH)
        features = np.concatenate((
            mfcc, zcr, rms, centroid, bandwidth, rolloff, contrast, chroma
        ), axis=0)

        aggregated_features = np.concatenate((
            mfcc.mean(axis=1),  # Mean across the time axis for MFCC
            [zcr.mean()],  # Mean of Zero Crossing Rate
            [rms.mean()],  # Mean of RMS Energy
            [centroid.mean()],  # Mean of Spectral Centroid
            [bandwidth.mean()],  # Mean of Spectral Bandwidth
            [rolloff.mean()],  # Mean of Spectral Roll-off
            contrast.mean(axis=1),  # Mean across the time axis for Spectral Contrast
            chroma.mean(axis=1)  # Mean across the time axis for Chroma
        ))

        return features, aggregated_features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
    





    # Feature column names
mfcc_cols = [f"mfcc{i+1}" for i in range(13)]

spec_cols = [
    "zcr",
    "rms",
    "centroid",
    "bandwidth",
    "rolloff"
]

contrast_cols = [f"spectral_contrast_band{i+1}" for i in range(7)] 

chroma_cols = [f"chroma{i+1}" for i in range(12)]

all_feature_names = mfcc_cols + spec_cols + contrast_cols + chroma_cols

features = []
aggregated_features = []
filenames = []
emotions = []
i = 0

for file in os.listdir(preprocessed_dataset_path):
    if file.lower().endswith(".wav"):
        path = os.path.join(preprocessed_dataset_path, file)
        result, aggregated_result = extract_features_from_audio(path)
        if result is not None and aggregated_result is not None:
            parts = file.split('_')
            emotion = parts[2]
            print(f"features are extracted from : {file}")
            feature_vector = [list(feature) for feature in result]
            features.append(feature_vector)
            aggregated_features.append(aggregated_result)
            filenames.append(file)
            emotions.append(emotion)

encoder = LabelEncoder()
encoded_emotions = encoder.fit_transform(emotions)

print("started writing features to csv file..")
df_framewise_features = pd.DataFrame(features, columns=all_feature_names)
df_framewise_features.insert(0, "file_name", filenames)
df_framewise_features['emotion'] = encoded_emotions
df_framewise_features.to_csv(csv_features_path, index=False)

print("started writing aggregated features to csv file..")
df_aggregated_features = pd.DataFrame(aggregated_features, columns=all_feature_names)
df_aggregated_features.insert(0, "file_name", filenames)
df_aggregated_features['emotion'] = encoded_emotions
df_aggregated_features.to_csv(csv_aggregated_features_path, index=False)








# Define features and target
X = df_aggregated_features.iloc[:, 1:-1]  # Features: all columns except first (filename) and last (target)
y = df_aggregated_features.iloc[:, -1]    # Target: last column

# Compute Mutual Information
mutual_info = mutual_info_classif(X, y)

# Compute Correlation (Pearson)
correlations = X.corrwith(y)

# Combine into a single DataFrame
aggregated_MI_corr_df = pd.DataFrame({ 
    "Feature": X.columns,
    "Mutual Information": mutual_info,
    "Correlation": correlations
})

aggregated_MI_corr_df.to_csv("Analytics/mi_corr_aggregated_features.csv", index=False)


feature_columns = df_framewise_features.columns[1:-1]
results = []

# Loop through each feature
for feature_name in feature_columns:
    # Parse the list into 2D array
    feature_series = df_framewise_features[feature_name].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    X = np.stack(feature_series.values)

    # Create frame-wise DataFrame
    frame_df = pd.DataFrame(X, columns=[f"{feature_name}_frame_{i}" for i in range(X.shape[1])])

    # Pearson correlation
    corr = frame_df.corrwith(y)

    # Mutual information
    mi = mutual_info_classif(frame_df, y, discrete_features=False)

    # Combine into result list
    for i in range(X.shape[1]):
        results.append({
            "Feature": feature_name,
            "Frame": i,
            "Frame_Name": f"{feature_name}_frame_{i}",
            "Correlation": corr.iloc[i],
            "Mutual Information": mi[i]
        })

# Combine into a single DataFrame
framewise_MI_corr_df = pd.DataFrame(results)

framewise_MI_corr_df.to_csv("Analytics/framewise_mi_corr_all_features.csv", index=False)







final_df = pd.DataFrame()

# Loop through all features
for feature in all_feature_names:
    # Get aggregated MI
    agg_mi = aggregated_MI_corr_df.loc[aggregated_MI_corr_df["Feature"] == feature, "Mutual Information"].values[0]
    
    # Get max framewise MI
    max_frame_mi = framewise_MI_corr_df[framewise_MI_corr_df["Feature"] == feature]["Mutual Information"].max()

    # Choose version with higher MI
    if max_frame_mi > agg_mi:
        print(feature)
        final_df[feature] = df_framewise_features[feature]
    else:
        final_df[feature] = df_aggregated_features[feature]

final_df.insert(0, "file_name", df_framewise_features["file_name"])
final_df["emotion"] = df_framewise_features["emotion"]

final_df.to_csv(csv_final_features_path, index=False)

print("Final features saved")


