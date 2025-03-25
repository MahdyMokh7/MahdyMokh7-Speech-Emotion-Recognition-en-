import os
import librosa
import matplotlib.pyplot as plt
import librosa.display

def plot_waveform(file_path, sr=16000):
    # Load audio
    y, sr = librosa.load(file_path, sr=sr)
    
    # Plot waveform
    plt.figure(figsize=(14, 4))
    librosa.display.waveshow(y, sr=sr, alpha=0.7)
    plt.title(f"Waveform of {file_path}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()

# Example usage:
plot_waveform(os.path.join("..", "PreProcessedDataSet_for_ML", "1001_DFA_ANG_XX.wav"))
