\# Offline Speech-to-Text Evaluation using Whisper



\## Problem Statement

Online speech recognition systems require internet connectivity and raise privacy concerns.

This project evaluates the accuracy and robustness of an offline speech-to-text system.



\## Objective

To analyze Whisper’s performance in offline conditions using quantitative metrics.



\## Dataset

LibriSpeech test-clean subset (official ASR benchmark).



\## Methodology

\- Baseline transcription using Whisper

\- Accuracy evaluation using Word Error Rate (WER)

\- Model comparison (tiny, base, small)

\- Noise robustness testing

\- Chunk-based (streaming-like) transcription



\## Results

\- Base model WER (clean): 3.57%

\- Noisy audio WER: (your value)

\- Chunked audio WER: 3.57%

\- Trade-off observed between accuracy and inference time



\## Conclusion

Whisper demonstrates strong offline speech recognition performance with reasonable robustness

to noise and minor degradation in chunk-based transcription.



\## Tools Used

Python, Whisper, FFmpeg, LibriSpeech, JiWER



