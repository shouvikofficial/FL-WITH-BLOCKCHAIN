from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="binary")
    return acc, f1
