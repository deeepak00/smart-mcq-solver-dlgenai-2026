import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

OPTIONS   = ['A', 'B', 'C', 'D', 'E']
LABEL2IDX = {opt: idx for idx, opt in enumerate(OPTIONS)}
IDX2LABEL = {idx: opt for idx, opt in enumerate(OPTIONS)}

class BpeTokenizerScratch:
    """
    Fits a Byte-Pair Encoding (BPE) subword tokenizer from scratch on training data.
    This resolves the Out-of-Vocabulary (UNK) issue with technical scientific terms
    without violating 'from-scratch' constraints.
    """
    def __init__(self, max_vocab: int = 20000):
        from tokenizers import Tokenizer as HFTokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import Whitespace
        
        self.tokenizer = HFTokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = Whitespace()
        self.max_vocab = max_vocab
        self.size = 4  # Initial size based on special tokens

    def build(self, texts: list[str]):
        """Trains the BPE tokenizer from an iterator of raw texts."""
        from tokenizers.trainers import BpeTrainer
        trainer = BpeTrainer(
            vocab_size=self.max_vocab,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
        )
        self.tokenizer.train_from_iterator(texts, trainer)
        self.size = self.tokenizer.get_vocab_size()

    def save(self, path: str):
        """Saves the HF tokenizer to a file."""
        self.tokenizer.save(path)

    def load(self, path: str):
        """Loads the HF tokenizer from a saved file."""
        from tokenizers import Tokenizer as HFTokenizer
        self.tokenizer = HFTokenizer.from_file(path)
        self.size = self.tokenizer.get_vocab_size()

    def encode(self, a: str, b: str, max_len: int = 128):
        """
        Tokenizes and pads prompt 'a' and option 'b' into [CLS] + a + [SEP] + b + [SEP].
        Employs smart truncation to preserve the prompt context.
        """
        ta = self.tokenizer.encode(str(a)).ids
        tb = self.tokenizer.encode(str(b)).ids
        
        # [CLS] ta [SEP] tb [SEP] takes 3 special tokens
        total_avail = max_len - 3
        if len(ta) + len(tb) > total_avail:
            # Keep option (tb) up to 48 tokens; allocate remainder to prompt (ta)
            opt_len = min(48, len(tb))
            ta = ta[:total_avail - opt_len]
            tb = tb[:total_avail - len(ta)]
            
        ids = [2] + ta + [3] + tb + [3]
        tids = [0] * (len(ta) + 2) + [1] * (len(tb) + 1)
        pad = max_len - len(ids)
        
        return ids + [0] * pad, tids + [0] * pad, [1] * len(ids) + [0] * pad

def texts_from(df: pd.DataFrame) -> list[str]:
    """Flattens prompts and option texts to fit the tokenizer."""
    return [str(val) for col in ['prompt'] + OPTIONS for val in df[col].tolist()]

def precompute(df: pd.DataFrame, tok: BpeTokenizerScratch, max_len: int, has_labels: bool = True):
    """Encodes the dataset once before training, speeding up the data loader loop."""
    n = len(df)
    ids_arr  = np.zeros((n, 5, max_len), dtype=np.int64)
    tids_arr = np.zeros((n, 5, max_len), dtype=np.int64)
    mask_arr = np.zeros((n, 5, max_len), dtype=np.int64)

    prompts = df['prompt'].astype(str).tolist()
    opt_cols = [df[o].astype(str).tolist() for o in OPTIONS]

    for i in range(n):
        p = prompts[i]
        for j in range(5):
            ids, tids, mask = tok.encode(p, opt_cols[j][i], max_len)
            ids_arr[i, j]  = ids
            tids_arr[i, j] = tids
            mask_arr[i, j] = mask

    labels = None
    if has_labels:
        labels = df['answer'].map(LABEL2IDX).values.astype(np.int64)
        
    return ids_arr, tids_arr, mask_arr, labels

class MCQTensorDataset(Dataset):
    """Loads pre-tokenized numpy arrays directly without overhead."""
    def __init__(self, ids, tids, mask, labels=None):
        self.ids = ids
        self.tids = tids
        self.mask = mask
        self.labels = labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        item = {
            'input_ids'     : torch.from_numpy(self.ids[i]),
            'token_type_ids': torch.from_numpy(self.tids[i]),
            'attention_mask': torch.from_numpy(self.mask[i]),
        }
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[i])
        return item
