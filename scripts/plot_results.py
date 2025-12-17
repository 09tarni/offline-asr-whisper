import pandas as pd
import matplotlib.pyplot as plt

# Load model comparison results
df = pd.read_csv("results/model_comparison.csv")

plt.figure()
plt.plot(df["Model"], df["WER (%)"], marker="o")
plt.xlabel("Whisper Model")
plt.ylabel("WER (%)")
plt.title("Model Size vs Word Error Rate")
plt.grid(True)
plt.savefig("plots/model_vs_wer.png")
plt.show()
