import numpy as np

OPTIONS = ['A', 'B', 'C', 'D', 'E']

def apk(actual, predicted, k=3):
    if len(predicted) > k:
        predicted = predicted[:k]

    for i, p in enumerate(predicted):
        if p == actual:
            return 1.0 / (i + 1)

    return 0.0


def mapk(actuals, predictions, k=3):
    return np.mean([apk(a, p, k) for a, p in zip(actuals, predictions)])