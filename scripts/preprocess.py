import os
import librosa
import numpy as np
import soundfile as sf
import scipy.signal
import noisereduce as nr
from pydub import AudioSegment
import librosa.display
import matplotlib.pyplot as plt


# === CONFIGURATION ===
MAIN_FOLDER_PATH = ".."
PRE_PROCESSED_PATH = ".."
DATASET_PATH = os.path.join(MAIN_FOLDER_PATH, "DataSet")
OUTPUT_FOLDER = os.path.join(PRE_PROCESSED_PATH, "PreProcessedDataSet_for_ML")
TARGET_SR = 16000   # sapmle rate
SILENCE_TOP_DB = 35
HIGHPASS_CUTOFF = 85  # cutoff to remove low-frequency noise
VISUALIZATION_DIR = "AudioVisualizations"
N_FFT = 512 # 32 ms window 
HOP_LENGTH = 205 # 12.8 ms step → 60% overlap

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)





def load_and_resample_audio(file_path, sr=TARGET_SR):
    """Load audio file and resample."""
    audio, sr = librosa.load(file_path, sr=sr)
    return audio, sr

def convert_to_mono(audio):
    """Convert stereo audio to mono."""
    return librosa.to_mono(audio)

def trim_silence(audio, top_db=SILENCE_TOP_DB):
    """Trim leading and trailing silence (if any)."""
    trimmed_audio, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed_audio

def normalize_audio(audio):
    """Normalize audio amplitude to peak value of 1."""
    return audio / np.max(np.abs(audio))

def apply_highpass_filter(audio, sr, cutoff=HIGHPASS_CUTOFF):
    """Apply high-pass filter to remove low-frequency noise."""
    sos = scipy.signal.butter(10, cutoff, btype='highpass', fs=sr, output='sos')
    return scipy.signal.sosfilt(sos, audio)

def reduce_noise(audio, sr):
    """Perform noise reduction using spectral gating."""
    return nr.reduce_noise(y=audio, sr=sr)

def is_not_empty(audio):
    if np.max(np.abs(audio)) < 1e-5:
        return 0
    return 1





SEGMENT_DURATION = 0.8
OVERLAP = 0.4

def split_audio(audio, sr, base_filename,
                                         segment_duration, overlap):
    segment_samples = int(sr * segment_duration)
    min_remainder_samples = int(sr * 0.1)

    i = 0
    start = 0
    while start + segment_samples <= len(audio):
        end = start + segment_samples
        chunk = audio[start:end]

        chunk_filename = f"{base_filename}_chunk{i+1}.wav"
        chunk_path = os.path.join(OUTPUT_FOLDER, chunk_filename)
        sf.write(chunk_path, chunk, sr)

        i += 1
        start = end - int(sr * overlap)   

    # Handle last remaining audio (if exist)
    remainder = len(audio) - (start + int(sr * overlap))
    if remainder >= min_remainder_samples:
        chunk = audio[start:]
        padding = segment_samples - len(chunk)
        padded_chunk = np.pad(chunk, (0, padding), mode='constant')
        
        #padded_chunk = normalize_audio(padded_chunk)

        chunk_filename = f"{base_filename}_chunk{i+1}.wav"
        chunk_path = os.path.join(OUTPUT_FOLDER, chunk_filename)
        sf.write(chunk_path, padded_chunk, sr)





        # =================== PIPELINE PROCESS FUNCTION ===================
def process_audio_file(file_path):
    """Apply full preprocessing pipeline to a single audio file (for traditional ML)."""
    try:
        print(f"Processing: {file_path}")
        audio, sr = load_and_resample_audio(file_path) 
        if(is_not_empty(audio)):
            audio = convert_to_mono(audio)
            audio = apply_highpass_filter(audio, sr)
            audio = reduce_noise(audio, sr)
            audio = trim_silence(audio)
            audio = normalize_audio(audio)

            split_audio(audio, sr, os.path.basename(file_path), SEGMENT_DURATION, OVERLAP)

    except Exception as error:
        print(f"Error processing {file_path}: {error}")






# =================== MAIN PIPELINE ===================
def preprocess_dataset():
    """Process all audio files in the dataset (traditional ML pipeline)."""
    for filename in os.listdir(DATASET_PATH):
        if filename.lower().endswith(".wav"):
            file_path = os.path.join(DATASET_PATH, filename)
            process_audio_file(file_path)
    print("\nAll audio files have been preprocessed and saved in: ", OUTPUT_FOLDER)


    

preprocess_dataset()