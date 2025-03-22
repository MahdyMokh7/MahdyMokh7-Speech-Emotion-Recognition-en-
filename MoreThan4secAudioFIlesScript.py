import os
import wave

def count_long_audio_files(directory, thresholds=[3, 3.25, 3.5, 3.75, 4]):
    results = {t: 0 for t in thresholds}
    total_files = 0

    for file in os.listdir(directory):
        if file.endswith(".wav"):
            total_files += 1
            filepath = os.path.join(directory, file)
            with wave.open(filepath, 'rb') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / float(rate)

                for t in thresholds:
                    if duration >= t:
                        results[t] += 1

    results_file_path = os.path.join(".", "DifferetnSecondsAnalyticsResults.txt")
    with open(results_file_path, "w") as f:
        f.write(f"Total WAV audio files: {total_files}\n")
        for t in thresholds:
            f.write(f"Audio files >= {t} seconds: {results[t]}\n")

    print(f"Results saved to {results_file_path}")

# Example usage
directory_path = os.path.join("..", "DataSet")
count_long_audio_files(directory_path)
