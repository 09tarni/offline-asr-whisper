import whisper
import time
import csv
from jiwer import wer
import string

def normalize(text):
    text = text.lower()
    text = text.replace("-", " ")
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text

with open("data/ground_truth.txt", "r", encoding="utf-8") as f:
    ground_truth = normalize(f.read())

models = ["tiny.en", "base.en", "small.en"]

with open("results/model_comparison.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Model", "Time (seconds)", "WER (%)"])

    for m in models:
        print(f"\nRunning model: {m}")
        model = whisper.load_model(m)

        start = time.time()
        result = model.transcribe("data/audio_clean.wav")
        end = time.time()

        prediction = normalize(result["text"])
        error = wer(ground_truth, prediction) * 100

        writer.writerow([m, round(end - start, 2), round(error, 2)])

        print(f"Time: {round(end - start, 2)} seconds")
        print(f"WER: {round(error, 2)} %")
