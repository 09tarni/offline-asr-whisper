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

with open("results/whisper_output.txt", "r", encoding="utf-8") as f:
    prediction = normalize(f.read())

error = wer(ground_truth, prediction)

print("Normalized Ground Truth:")
print(ground_truth)
print("\nNormalized Prediction:")
print(prediction)
print("\nWord Error Rate (WER):", round(error * 100, 2), "%")
