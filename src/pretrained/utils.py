import os
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, f1_score
from transformers import logging as hf_logging

# ================= CONFIG (unchanged from the original notebook) =================
MODEL_NAME = "roberta-large"
FOLDS = 5
MAX_LEN = 386
SEED = 42

LABEL_MAP = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
INV_MAP = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
OPTION_COLS = ["A", "B", "C", "D", "E"]


def silence_warnings():
    """Suppresses the noisy-but-harmless HF loading/key-mismatch warnings."""
    hf_logging.set_verbosity_error()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    print("DEVICE:", device, "| bf16 supported:", use_bf16)
    return device, use_bf16


def map_at_3(labels, logits):
    """Mean Average Precision @ 3 - matches the competition's top-3 submission format."""
    top3 = np.argsort(-logits, axis=1)[:, :3]
    score = 0.0
    for i, label in enumerate(labels):
        if label == top3[i, 0]:
            score += 1.0
        elif label == top3[i, 1]:
            score += 1.0 / 2
        elif label == top3[i, 2]:
            score += 1.0 / 3
    return score / len(labels)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "map3": map_at_3(labels, logits),
        "precision": precision_score(labels, preds, average="weighted", zero_division=0),
        "f1": f1_score(labels, preds, average="weighted", zero_division=0),
    }