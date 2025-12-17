from jiwer import wer
import string

def normalize(text):
    text = text.lower()
    text = text.replace("-", " ")
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text

with open("data/ground_truth.txt", "r", encoding="utf-8") as f:
    gt = normalize(f.read())

with open("results/whisper_chunked_output.txt", "r", encoding="utf-8") as f:
    pred = normalize(f.read())

error = wer(gt, pred)

print("Chunked Audio WER:", round(error * 100, 2), "%")
