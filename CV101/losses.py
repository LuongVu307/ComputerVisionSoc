import numpy as np


class MeanSquaredError:
    def __init__(self):
        self.y_pred = None
        self.y_true = None
        self.loss = None

    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        self.loss = np.mean((y_pred - y_true) ** 2)
        return self.loss

    def backward(self):
        n = len(self.y_true)
        return (2 / n) * (self.y_pred - self.y_true)



class CategoricalCrossEntropy:
    def __init__(self):
        self.y_pred = None
        self.y_true = None
        self.loss = None

    def forward(self, y_pred, y_true):
        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        self.y_pred = y_pred
        self.y_true = y_true
        self.loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        return self.loss

    def backward(self):
        n = len(self.y_true)
        grad = -self.y_true / self.y_pred
        return grad / n

class BinaryCrossEntropy:
    def __init__(self):
        self.y_pred = None
        self.y_true = None
        self.loss = None

    def forward(self, y_pred, y_true):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        self.y_pred = y_pred
        self.y_true = y_true
        self.loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return self.loss

    def backward(self):
        n = len(self.y_true)
        grad = (self.y_pred - self.y_true) / (self.y_pred * (1 - self.y_pred))
        return grad / n
