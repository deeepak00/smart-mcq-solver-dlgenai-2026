import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, x, mask):
        scores = self.attn(x).squeeze(-1)
        fill_value = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(mask == 0, fill_value)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (x * weights).sum(dim=1)


class EncoderBlock(nn.Module):
    def __init__(self, vocab_size, d_model=256, conv_ch=256, hidden=256, drop=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.conv1 = nn.Conv1d(d_model, conv_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(conv_ch, conv_ch, kernel_size=5, padding=2)
        self.bigru = nn.GRU(conv_ch, hidden, batch_first=True, bidirectional=True)
        self.pool = AttentionPooling(hidden * 2)
        self.drop = nn.Dropout(drop)

    def forward(self, ids, mask):
        x = self.emb(ids)              # [B, T, D]
        x = x.transpose(1, 2)          # [B, D, T]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.transpose(1, 2)          # [B, T, C]
        x, _ = self.bigru(x)           # [B, T, 2H]
        out = self.pool(x, mask)
        return self.drop(out)          # [B, 2H]


class ScratchMCQModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, conv_ch=256, hidden=256, drop=0.2):
        super().__init__()
        self.encoder = EncoderBlock(vocab_size, d_model, conv_ch, hidden, drop)

        feat_dim = hidden * 2 * 4  
        self.head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(feat_dim // 2, feat_dim // 4),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(feat_dim // 4, 1)
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
                if m.padding_idx is not None:
                    with torch.no_grad():
                        m.weight[m.padding_idx].fill_(0.0)

    def forward(self, input_ids, token_type_ids, attention_mask):
        B, N, T = input_ids.shape

        ids = input_ids.reshape(B * N, T)
        mask = attention_mask.reshape(B * N, T)

        q_repr = self.encoder(ids, mask)
        o_repr = q_repr

        q_repr = q_repr.view(B, N, -1)
        o_repr = o_repr.view(B, N, -1)

        fused = torch.cat([
            q_repr,
            o_repr,
            torch.abs(q_repr - o_repr),
            q_repr * o_repr
        ], dim=-1)

        logits = self.head(fused).squeeze(-1)
        return logits