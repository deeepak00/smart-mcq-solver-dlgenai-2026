import random
import math
import numpy as np
import pandas as pd
import torch

def seed_everything(seed=42):
    """Sets random seeds for reproducibility across systems."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def apk(actual, predicted, k=3):
    """Computes Average Precision at k (used by original code)."""
    predicted = predicted[:k]
    for i, pred in enumerate(predicted):
        if pred == actual:
            return 1.0 / (i + 1)
    return 0.0

def mapk(actuals, predictions, k=3):
    """Computes Mean Average Precision at k (used by original code)."""
    scores = [
        apk(actual, pred, k)
        for actual, pred in zip(actuals, predictions)
    ]
    return np.mean(scores)

def save_submission(test_ids, predictions, path="submission.csv"):
    """Saves predictions to a CSV file (used by original code)."""
    submission = pd.DataFrame({
        "ID": test_ids,
        "Prediction": predictions
    })
    submission.to_csv(path, index=False)
    print(f"Submission saved to {path}")

