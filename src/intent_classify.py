# src/intent_classify.py

import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
import joblib

# Where we save the trained classifier
MODEL_PATH = Path(__file__).resolve().parent / "intent_clf.joblib"


class IntentClassifier:
    def __init__(self):
        self.model = None
        # Load existing model if it exists
        if MODEL_PATH.exists():
            self.model = joblib.load(MODEL_PATH)

    def train(self, X, y):
        """
        Train a simple logistic regression intent classifier on feature vectors.
        X: list of feature arrays
        y: list of integer labels
        """
        X = np.stack(X)          # (N, D)
        y = np.array(y)          # (N,)

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)

        self.model = clf
        joblib.dump(self.model, MODEL_PATH)
        print(f"[IntentClassifier] Trained and saved to {MODEL_PATH}")

    def predict(self, feat):
        """
        Predict intent label for a single feature vector.
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet. Run training first.")

        feat = np.asarray(feat).reshape(1, -1)
        pred = self.model.predict(feat)[0]
        return str(pred)

