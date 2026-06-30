import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinPE(nn.Module):
    """Sinusoidal Positional Encoding."""
    def __init__(self, d: int, max_len: int = 512):
        super().__init__()
        pe  = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention layer with key-padding mask support."""
    def __init__(self, d: int, h: int, drop: float = 0.1):
        super().__init__()
        assert d % h == 0, "d must be divisible by h"
        self.h, self.dh = h, d // h
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x, mask=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        
        split = lambda t: t.view(B, T, self.h, self.dh).transpose(1, 2)
        q, k, v = split(q), split(k), split(v)
        
        s = (q @ k.transpose(-2, -1)) / (self.dh ** 0.5)
        
        if mask is not None:
            # Broadcast mask (B*5, T) -> (B*5, 1, 1, T) to hide padding key tokens
            s = s.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
            
        a = self.drop(F.softmax(s, dim=-1))
        return self.out((a @ v).transpose(1, 2).reshape(B, T, C))


class TransformerBlock(nn.Module):
    """Standard Pre-LN Transformer encoder block."""
    def __init__(self, d: int, h: int, ff: int, drop: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.attn = MultiHeadAttention(d, h, drop)
        self.ff = nn.Sequential(
            nn.Linear(d, ff),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(ff, d),
            nn.Dropout(drop)
        )
        
    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        return x + self.ff(self.ln2(x))


class ScratchMCQTransformer(nn.Module):
    """
    Scratch MCQ transformer architecture. Runs parallel evaluation
    over all 5 options, returning relative logits for the labels.
    """
    def __init__(self, vocab: int, d: int = 256, h: int = 8, layers: int = 4, ff: int = 512, drop: float = 0.15):
        super().__init__()
        self.emb  = nn.Embedding(vocab, d, padding_idx=0)
        self.temb = nn.Embedding(2, d)
        self.pe   = SinPE(d)
        self.enc  = nn.ModuleList([TransformerBlock(d, h, ff, drop) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(d // 2, 1)
        )
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.02)
                # CRITICAL: Keep embedding weights of padding token set to 0.0
                if m.padding_idx is not None:
                    with torch.no_grad():
                        m.weight[m.padding_idx].fill_(0.0)

    def encode(self, ids, tids, mask):
        x = self.pe(self.emb(ids) + self.temb(tids))
        for blk in self.enc:
            x = blk(x, mask)
        return self.norm(x)[:, 0]  # Extracts CLS representation

    def forward(self, input_ids, token_type_ids, attention_mask):
        B, N, T = input_ids.shape
        # Flatten option batch dimension to forward concurrently
        cls = self.encode(
            input_ids.reshape(B * N, T),
            token_type_ids.reshape(B * N, T),
            attention_mask.reshape(B * N, T)
        ).view(B, N, -1)
        return self.head(cls).squeeze(-1)
