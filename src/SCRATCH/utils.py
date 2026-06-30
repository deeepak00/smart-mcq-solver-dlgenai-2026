import math
import numpy as np
import torch

def set_seed(seed: int = 42):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def softmax_np(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)

def map3(y_true, probs):
    scores = []
    for yt, yp in zip(y_true, probs):
        top3 = np.argsort(yp)[::-1][:3]
        s, h = 0., 0
        for i, p in enumerate(top3):
            if p == yt:
                h += 1
                s += h / (i + 1)
        scores.append(s)
    return np.mean(scores)

def warmup_cosine(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))
