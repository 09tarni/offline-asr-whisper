import whisper
import time
import os

model = whisper.load_model("base")

audio_path = "data/audio_clean.wav"
output_path = "results/whisper_output.txt"

start = time.time()
result = model.transcribe(audio_path)
end = time.time()

os.makedirs("results", exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    f.write(result["text"])

print("Transcription saved to:", output_path)
print("Time taken (seconds):", round(end - start, 2))
