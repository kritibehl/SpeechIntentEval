import os
import json
from pathlib import Path

from .preprocess import extract_features
from .intent_classify import IntentClassifier

AUDIO_DIR = Path(__file__).resolve().parents[1] / "audio_samples"
LABELS_PATH = Path(__file__).resolve().parent / "labels.json"


def main():
    if not AUDIO_DIR.exists():
        raise FileNotFoundError(f"Audio directory not found: {AUDIO_DIR}")

    with open(LABELS_PATH) as f:
        labels = json.load(f)  # e.g., {"STOP_INTENT": 0, "GO_INTENT": 1, ...}

    X, y = [], []
    file_names = []

    # Build training set from all WAVs in audio_samples/
    for fname in os.listdir(AUDIO_DIR):
        if not fname.endswith(".wav"):
            continue

        path = AUDIO_DIR / fname
        intent_name = fname.split(".")[0].upper() + "_INTENT"  # stop.wav → STOP_INTENT

        if intent_name not in labels:
            print(f"[WARN] No label found for {intent_name} (file {fname}), skipping.")
            continue

        feat = extract_features(str(path))
        X.append(feat)
        y.append(labels[intent_name])
        file_names.append(fname)

    if not X:
        print("[ERROR] No valid WAV files found in audio_samples/.")
        return

    clf = IntentClassifier()
    clf.train(X, y)

    print(f"[IntentClassifier] Trained on {len(X)} examples")
    for fname, feat in zip(file_names, X):
        pred_id = clf.predict(feat)
        # Inverse label lookup
        inv_labels = {v: k for k, v in labels.items()}
        pred_name = inv_labels.get(int(pred_id), f"UNKNOWN({pred_id})")
        print(f"{fname} → {pred_name}")


if __name__ == "__main__":
    main()
