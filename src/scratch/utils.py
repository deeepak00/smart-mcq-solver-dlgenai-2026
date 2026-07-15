import math
import warnings
import numpy as np
import torch
from sklearn.metrics import precision_score, f1_score
warnings.filterwarnings('ignore')


OPTIONS   = ['A', 'B', 'C', 'D', 'E']
LABEL2IDX = {opt: idx for idx, opt in enumerate(OPTIONS)}
IDX2LABEL = {idx: opt for idx, opt in enumerate(OPTIONS)}


CFG = {
    'folds': 5,
    'epochs': 8,
    'bs': 24,
    'lr': 2.5e-4,
    'wd': 0.01,
    'd_model': 256,
    'conv_ch': 256,
    'hidden': 256,
    'max_len': 128,
    'drop': 0.2,
    'patience': 3,
    'label_smoothing': 0.08,
    'warmup_frac': 0.1,
    'max_vocab': 30000,
    'seed': 42,
}



def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = (device.type == 'cuda')

    if use_amp:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU detected — running on CPU.")

    return device, use_amp



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


def compute_metrics(y_true, logits):
    probs = softmax_np(logits)
    preds = logits.argmax(1)

    return {
        'accuracy': float((preds == y_true).mean()),
        'map3': float(map3(y_true, probs)),
        'precision_macro': float(precision_score(y_true, preds, average='macro', zero_division=0)),
        'f1_macro': float(f1_score(y_true, preds, average='macro', zero_division=0)),
    }