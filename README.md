# 🎙️ SpeechIntentEval — Tiny Speech Command Intent Evaluator

> **Goal:** Evaluate how reliably simple speech commands are recognized and mapped to *intents* (e.g., “stop”, “go”, “turn on/off”).

**SpeechIntentEval** is a small, self-contained project that turns a handful of short speech commands into **discrete intents** and evaluates how consistently they are classified.

This is designed as a *research-style mini-benchmark* you can extend with:
- More commands  
- Different feature extractors  
- Stronger classifiers or end-to-end speech models  

---

## 🔎 What this project does

Given a few WAV files like:

- `stop.wav` → `STOP_INTENT`  
- `go.wav` → `GO_INTENT`  
- `on.wav` → `TURN_ON_INTENT`  
- `off.wav` → `TURN_OFF_INTENT`  

the pipeline:

1. **Loads audio** (`librosa`)  
2. **Extracts features** (e.g., MFCCs)  
3. **Trains a tiny intent classifier** (`scikit-learn`)  
4. **Predicts intents** for each file  
5. Optionally computes **accuracy & confusion matrix** over your labeled examples.

This gives a clean, reproducible baseline for **voice command understanding**.

---

## 🗂️ Project Structure

```text
SpeechIntentEval/
├── audio_samples/          # Small example WAV files (stop.wav, go.wav, on.wav, off.wav, ...)
├── src/
│   ├── __init__.py
│   ├── labels.json         # Mapping from intent name → numeric ID
│   ├── load_audio.py       # Load WAV with librosa
│   ├── preprocess.py       # Feature extraction (e.g., MFCCs)
│   ├── intent_classify.py  # IntentClassifier wrapper (fit / predict)
│   ├── run_eval.py         # End-to-end demo: train + print predictions
│   └── evaluate_intents.py # Simple evaluation: accuracy + confusion matrix
├── requirements.txt
└── README.md
