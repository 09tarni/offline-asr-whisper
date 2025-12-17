import whisper
import os

model = whisper.load_model("base")

chunk_files = sorted([f for f in os.listdir("data") if f.startswith("chunk_")])
full_text = ""

for chunk in chunk_files:
    print("Transcribing:", chunk)
    result = model.transcribe(os.path.join("data", chunk))
    full_text += result["text"] + " "

with open("results/whisper_chunked_output.txt", "w", encoding="utf-8") as f:
    f.write(full_text.strip())

print("Chunked transcription saved")
