import os
import pandas as pd
from pydub import AudioSegment
import ffmpeg
from pydub import AudioSegment

# Use ffmpeg-python as the backend
AudioSegment.converter = ffmpeg

dataset_folder = os.path.join( "..", "DataSet")
csv_file = os.path.join(".","crema_d_dataset.csv")

df = pd.read_csv(csv_file)

durations = []

for index, row in df.iterrows():
    audio_file_path = os.path.join(dataset_folder, row['File Name'])
    
    audio = AudioSegment.from_file(audio_file_path)
    
    duration_seconds = len(audio) / 1000.0

    durations.append(duration_seconds)

df['Duration'] = durations

df.to_csv(csv_file, index=False)

print("Durations added successfully!")