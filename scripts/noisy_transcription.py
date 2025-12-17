import whisper
import time

model = whisper.load_model("base")

start = time.time()
result = model.transcribe("data/audio_noisy.wav")
end = time.time()

with open("results/whisper_noisy_output.txt", "w", encoding="utf-8") as f:
    f.write(result["text"])

print("Noisy transcription done")
print("Time taken:", round(end - start, 2), "seconds")
