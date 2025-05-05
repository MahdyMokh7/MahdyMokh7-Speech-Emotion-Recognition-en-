FROM python:3.8-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# Ensure ffmpeg is installed for pydub (audio processing)
# RUN apt-get update && apt-get install -y ffmpeg

CMD ["python", "pipeline.py"]
