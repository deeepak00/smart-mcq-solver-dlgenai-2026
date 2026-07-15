import numpy as np
import torch
from torch.utils.data import Dataset
from utils import OPTIONS


class BpeTokenizerScratch:
    def __init__(self, max_vocab: int = 20000):
        from tokenizers import Tokenizer as HFTokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import Whitespace

        self.tokenizer = HFTokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = Whitespace()
        self.max_vocab = max_vocab
        self.size = 4

    def build(self, texts):
        from tokenizers.trainers import BpeTrainer
        trainer = BpeTrainer(
            vocab_size=self.max_vocab,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
        )
        self.tokenizer.train_from_iterator(texts, trainer)
        self.size = self.tokenizer.get_vocab_size()

    def encode_pair(self, prompt: str, option: str, max_len: int = 128):
        tp = self.tokenizer.encode(str(prompt)).ids
        to = self.tokenizer.encode(str(option)).ids

        total = max_len - 3
        if len(tp) + len(to) > total:
            keep_o = min(48, len(to))
            tp = tp[:total - keep_o]
            to = to[:total - len(tp)]

        ids = [2] + tp + [3] + to + [3]
        tids = [0] * (len(tp) + 2) + [1] * (len(to) + 1)
        mask = [1] * len(ids)

        pad = max_len - len(ids)
        if pad > 0:
            ids += [0] * pad
            tids += [0] * pad
            mask += [0] * pad

        return ids, tids, mask

    def save(self, path: str):
        self.tokenizer.save(path)

    @classmethod
    def load(cls, path: str):
        from tokenizers import Tokenizer as HFTokenizer
        obj = cls.__new__(cls)
        obj.tokenizer = HFTokenizer.from_file(path)
        obj.max_vocab = None
        obj.size = obj.tokenizer.get_vocab_size()
        return obj


def texts_from(df):
    return [str(v) for col in ['prompt'] + OPTIONS for v in df[col].astype(str).tolist()]


def precompute(df, tok, max_len, has_labels=True):
    n = len(df)
    ids_arr  = np.zeros((n, 5, max_len), dtype=np.int64)
    tids_arr = np.zeros((n, 5, max_len), dtype=np.int64)
    mask_arr = np.zeros((n, 5, max_len), dtype=np.int64)

    prompts = df['prompt'].astype(str).tolist()
    opts = [df[o].astype(str).tolist() for o in OPTIONS]

    for i in range(n):
        p = prompts[i]
        for j in range(5):
            ids, tids, mask = tok.encode_pair(p, opts[j][i], max_len)
            ids_arr[i, j] = ids
            tids_arr[i, j] = tids
            mask_arr[i, j] = mask

    labels = df['answer'].values.astype(np.int64) if has_labels else None
    return ids_arr, tids_arr, mask_arr, labels


class MCQTensorDataset(Dataset):
    def __init__(self, ids, tids, mask, labels=None):
        self.ids = ids
        self.tids = tids
        self.mask = mask
        self.labels = labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        item = {
            'input_ids': torch.from_numpy(self.ids[i]),
            'token_type_ids': torch.from_numpy(self.tids[i]),
            'attention_mask': torch.from_numpy(self.mask[i]),
        }
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[i])
        return item