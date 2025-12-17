import numpy as np
import soundfile as sf

audio, sr = sf.read("data/audio_clean.wav")

# Add white noise
noise = np.random.normal(0, 0.02, audio.shape)
noisy_audio = audio + noise

sf.write("data/audio_noisy.wav", noisy_audio, sr)

print("Noisy audio saved as data/audio_noisy.wav")
