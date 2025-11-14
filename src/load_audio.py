import librosa
import numpy as np

def load_audio(path, sr=16000):
    audio, sr = librosa.load(path, sr=sr)
    return audio, sr

# optional helper (if you'd like)
def load_wav(path):
    audio, sr = load_audio(path)
    return audio
