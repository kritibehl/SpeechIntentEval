import numpy as np
from sklearn.neighbors import KNeighborsClassifier

class IntentClassifier:
    def __init__(self):
        self.model = KNeighborsClassifier(n_neighbors=3)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, features):
        return self.model.predict([features])[0]
