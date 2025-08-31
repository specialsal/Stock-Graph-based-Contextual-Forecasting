# coding: utf-8
"""
Fusion Model：时间序列编码 + 行业/板块 Embedding + 截面交互注意力
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CFG


# =============== 1. 编码器 ===============
class DailyEncoder(nn.Module):
    def __init__(self, in_dim: int = 6, hidden: int = CFG.hidden):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.pe   = nn.Parameter(torch.randn(1, CFG.daily_window, hidden) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(hidden, nhead=4,
                                               dim_feedforward=hidden * 4,
                                               dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):                           # [B, Td, 6]
        h = self.proj(x) + self.pe
        h = self.encoder(h)
        h = self.norm(h)
        return h.mean(1)                            # [B, hidden]


class MinuteEncoder(nn.Module):
    def __init__(self, in_dim: int = 6, hidden: int = CFG.hidden):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, hidden // 2, kernel_size=5,
                               stride=2, padding=2)
        self.bn1   = nn.BatchNorm1d(hidden // 2)
        self.conv2 = nn.Conv1d(hidden // 2, hidden, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(hidden)
        self.gru   = nn.GRU(hidden, hidden, num_layers=1,
                             batch_first=True, bidirectional=False)
        self.attn  = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):                           # [B, Tm, 6]
        x = x.transpose(1, 2)                       # [B, 6, Tm]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))         # [B, H, T']
        x = x.transpose(1, 2)                       # [B, T', H]
        out, _ = self.gru(x)                        # [B, T', H]
        w = torch.softmax(self.attn(out), dim=1)    # [B, T', 1]
        return (out * w).sum(1)                     # [B, hidden]


# =============== 2. 主模型 ===============
class FusionModel(nn.Module):
    def __init__(self, num_industries: int, num_sectors: int):
        super().__init__()
        # --- 编码 ---
        self.daily_enc  = DailyEncoder()
        self.minute_enc = MinuteEncoder()

        self.ind_emb = nn.Embedding(num_industries, CFG.ind_emb_dim)
        self.sec_emb = nn.Embedding(num_sectors,   CFG.sec_emb_dim)

        # gate 融合日/分钟
        self.gate = nn.Linear(CFG.hidden * 2, 1)

        self.stk_dim = CFG.hidden + CFG.ind_emb_dim + CFG.sec_emb_dim
        self.ln_in   = nn.LayerNorm(self.stk_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.stk_dim) * 0.02)

        self.attn_r2s = nn.MultiheadAttention(self.stk_dim, CFG.attn_heads,
                                              dropout=0.1, batch_first=True)
        self.attn_s2r = nn.MultiheadAttention(self.stk_dim, CFG.attn_heads,
                                              dropout=0.1, batch_first=True)

        final_dim = self.stk_dim * 3
        self.head = nn.Sequential(
            nn.LayerNorm(final_dim),
            nn.Linear(final_dim, 256),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    # ---------- Encode ----------
    def encode(self, daily, minute, ind_id, sec_id):
        h_d   = self.daily_enc(daily)               # [B, H]
        h_m   = self.minute_enc(minute)             # [B, H]

        gate   = torch.sigmoid(self.gate(torch.cat([h_d, h_m], -1)))
        h_prc  = gate * h_d + (1. - gate) * h_m     # [B, H]

        h_ind = self.ind_emb(ind_id)                # [B, 32]
        h_sec = self.sec_emb(sec_id)                # [B, 32]

        stk = torch.cat([h_prc, h_ind, h_sec], -1)  # [B, D]
        return self.ln_in(stk)

    # ---------- Interact ----------
    def interact_and_head(self, stk_repr):
        B = stk_repr.size(0)
        stk_seq = stk_repr.unsqueeze(0)             # [1, B, D]

        # R: cls_token → stocks
        R, _ = self.attn_r2s(self.cls_token, stk_seq, stk_seq)

        # Stocks → R (cross)
        cross, _ = self.attn_s2r(stk_seq, R, R)

        stk_seq  = stk_seq.squeeze(0)               # [B, D]
        R_expand = R.squeeze(0).repeat(B, 1)        # [B, D]
        cross    = cross.squeeze(0)                 # [B, D]

        final = torch.cat([stk_seq, R_expand, cross], -1)
        return self.head(final).squeeze(-1)

    # ---------- forward ----------
    def forward(self, daily, minute, ind_id, sec_id):
        stk = self.encode(daily, minute, ind_id, sec_id)
        return self.interact_and_head(stk)