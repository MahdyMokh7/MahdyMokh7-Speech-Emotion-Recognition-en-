import librosa
import librosa.display
import matplotlib.pyplot as plt
import os

# Define input and output folder paths
audio_folder = os.path.join("sampleAudios")
output_folder = "sampleWaveforms"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through all files in the audio folder
for filename in os.listdir(audio_folder):
    if filename.endswith(".wav") or filename.endswith(".mp3"):
        file_path = os.path.join(audio_folder, filename)
        y, sr = librosa.load(file_path, sr=None)

        # Create a new figure for each audio file
        plt.figure(figsize=(6, 2))
        
        # Set background color to light tan
        plt.gcf().set_facecolor('#F5DEB3')  # light tan background
        
        # Plot the waveform with dark brown color
        librosa.display.waveshow(y, sr=sr, color='#8B4513')  # dark brown color

        # Hide the axes for a cleaner image
        plt.axis("off")
        
        # Generate output file path and save the image
        output_path = os.path.join(output_folder, filename.replace(".wav", ".png").replace(".mp3", ".png"))
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
        plt.close()

print("Waveform images saved successfully!")
