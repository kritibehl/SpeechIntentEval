import os
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .preprocess import extract_features
from .intent_classify import IntentClassifier

AUDIO_DIR = Path(__file__).resolve().parents[1] / "audio_samples"
LABELS_PATH = Path(__file__).resolve().parent / "labels.json"
OUT_CONFUSION = Path(__file__).resolve().parents[1] / "intent_confusion_matrix.csv"


def main():
    if not AUDIO_DIR.exists():
        raise FileNotFoundError(f"Audio directory not found: {AUDIO_DIR}")

    with open(LABELS_PATH) as f:
        labels = json.load(f)  # e.g., {"STOP_INTENT": 0, "GO_INTENT": 1, ...}
    inv_labels = {v: k for k, v in labels.items()}

    X, y_true, fnames = [], [], []

    for fname in os.listdir(AUDIO_DIR):
        if not fname.endswith(".wav"):
            continue

        path = AUDIO_DIR / fname
        intent_name = fname.split(".")[0].upper() + "_INTENT"

        if intent_name not in labels:
            print(f"[WARN] No label for {intent_name} (file {fname}), skipping.")
            continue

        feat = extract_features(str(path))
        X.append(feat)
        y_true.append(labels[intent_name])
        fnames.append(fname)

    if not X:
        print("[ERROR] No valid WAV files found in audio_samples/.")
        return

    clf = IntentClassifier()
    clf.train(X, y_true)

    y_pred = [int(clf.predict(f)) for f in X]

    acc = accuracy_score(y_true, y_pred)
    print(f"\n=== Intent Classification Metrics ===")
    print(f"Accuracy: {acc:.3f} on {len(y_true)} examples\n")

    target_names = [inv_labels[i] for i in sorted(inv_labels.keys())]
    print("Classification report:")
    print(
        classification_report(
            y_true, y_pred, labels=sorted(inv_labels.keys()), target_names=target_names
        )
    )

    cm = confusion_matrix(y_true, y_pred, labels=sorted(inv_labels.keys()))
    np.savetxt(OUT_CONFUSION, cm, fmt="%d", delimiter=",")
    print(f"\nConfusion matrix saved to: {OUT_CONFUSION}")


if __name__ == "__main__":
    main()
