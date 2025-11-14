import librosa
import numpy as np

def extract_features(wav_path, n_mfcc=40):
    y, sr = librosa.load(wav_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)
