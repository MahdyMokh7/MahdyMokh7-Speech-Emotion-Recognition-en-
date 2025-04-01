import librosa
import librosa.display
import matplotlib.pyplot as plt
import os

audio_folder = os.path.join("sample audios")
output_folder = "Waveforms"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(audio_folder):
    if filename.endswith(".wav") or filename.endswith(".mp3"):
        file_path = os.path.join(audio_folder, filename)
        y, sr = librosa.load(file_path, sr=None)

        plt.figure(figsize=(6, 2))
        
        librosa.display.waveshow(y, sr=sr, color='#D2B48C')  # light brown color

        plt.axis("off")
        
        output_path = os.path.join(output_folder, filename.replace(".wav", ".png").replace(".mp3", ".png"))
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
        plt.close()

print("Waveform images saved successfully!")
