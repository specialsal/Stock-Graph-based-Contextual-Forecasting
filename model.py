# coding: utf-8
"""
模型定义：多尺度时序编码 + 行业/板块 Embedding + 双阶段截面交互
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import CFG


# ================= 1. 时序编码器 =================
class DailyEncoder(nn.Module):
    """日线 Transformer 编码器"""
    def __init__(self, in_dim: int = 6, hidden: int = CFG.hidden):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)                       # 线性映射到 hidden
        self.pe   = nn.Parameter(torch.randn(1, CFG.daily_window, hidden) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=4, dim_feedforward=hidden * 4,
            dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):                                           # x:[B, Td, C]
        h = self.proj(x) + self.pe
        h = self.encoder(h)
        h = self.norm(h)
        return h.mean(dim=1)                                        # [B, hidden]


class MinuteEncoder(nn.Module):
    """30 分钟 Conv + GRU + Attention 编码器"""
    def __init__(self, in_dim: int = 6, hidden: int = CFG.hidden):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, hidden // 2, kernel_size=5, stride=2, padding=2)
        self.bn1   = nn.BatchNorm1d(hidden // 2)
        self.conv2 = nn.Conv1d(hidden // 2, hidden, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(hidden)

        self.gru = nn.GRU(
            input_size=hidden, hidden_size=hidden,
            num_layers=2, batch_first=True,
            bidirectional=True, dropout=0.1
        )

        self.attn = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):                                           # x:[B, Tm, C]
        x = x.transpose(1, 2)                                       # -> [B, C, Tm]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.transpose(1, 2)                                       # -> [B, T', C]

        out, _ = self.gru(x)                                        # [B, T', 2H]
        w = torch.softmax(self.attn(out), dim=1)                    # Attention 权重
        return (out * w).sum(dim=1)                                 # [B, 2H]


# ================= 2. 主模型 =================
class FusionModel(nn.Module):
    """
    * 股票自身时序特征  = 日线 + 30min
    * 离散标签          = 行业 + 板块
    * 截面交互          = 市场隐状态 R → 截面 B → 反馈到个股
    """
    def __init__(self,
                 num_industries: int = 100,
                 num_sectors: int = 500):                           # 数量可在 dataset 中动态传入
        super().__init__()

        # ---------- 编码分支 ----------
        self.daily_enc  = DailyEncoder()
        self.minute_enc = MinuteEncoder()

        # 行业 / 板块 Embedding
        self.ind_emb = nn.Embedding(num_industries, CFG.ind_emb_dim)
        self.sec_emb = nn.Embedding(num_sectors,   CFG.sec_emb_dim)

        # 原始个股表征维度
        self.stk_dim = CFG.hidden + CFG.hidden * 2 + CFG.ind_emb_dim + CFG.sec_emb_dim
        self.ln_in   = nn.LayerNorm(self.stk_dim)

        # learnable cls_token 用于抽取市场隐状态 R
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.stk_dim))

        # 双阶段多头注意力
        self.attn1 = nn.MultiheadAttention(self.stk_dim, num_heads=8, batch_first=True, dropout=0.1)
        self.attn2 = nn.MultiheadAttention(self.stk_dim, num_heads=8, batch_first=True, dropout=0.1)

        # 输出层（拼接三个向量：stk_repr | R | cross）
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

    # -------------------------------------------------
    def forward(self,
                daily:  torch.Tensor,                               # [B,Td,6]
                minute: torch.Tensor,                               # [B,Tm,6]
                ind_id: torch.Tensor,                               # [B]   int
                sec_id: torch.Tensor):                              # [B]   int
        """前向传播"""
        # 1) 个股时序编码
        h_d = self.daily_enc(daily)                                 # [B,128]
        h_m = self.minute_enc(minute)                               # [B,256]

        # 2) 行业 / 板块 Embedding
        h_ind = self.ind_emb(ind_id)                                # [B,32]
        h_sec = self.sec_emb(sec_id)                                # [B,32]

        # 3) 拼接成原始股票表征
        stk_repr = torch.cat([h_d, h_m, h_ind, h_sec], dim=-1)      # [B,stk_dim]
        stk_repr = self.ln_in(stk_repr).unsqueeze(0)                # [1,B,D] —— 视作序列

        # 4) 抽取市场隐状态 R（cls_token + Self Attn）
        seq_with_cls = torch.cat([self.cls_token, stk_repr], dim=1) # [1,B+1,D]
        seq_out, _   = self.attn1(seq_with_cls, seq_with_cls, seq_with_cls)
        R = seq_out[:, 0:1, :]                                      # 取 cls 位置 → [1,1,D]

        # 5) 第一次交叉注意力：R → Stocks，生成 B
        B, _ = self.attn1(R, stk_repr, stk_repr)                    # [1,1,D]

        # 6) 第二次交叉注意力：Stocks → B
        B_expanded = B.repeat(1, stk_repr.size(1), 1)               # [1,B,D]
        cross_out, _ = self.attn2(stk_repr, B_expanded, B_expanded) # [1,B,D]

        # 7) 拼接三个向量
        stk_repr = stk_repr.squeeze(0)      # [B,D]
        cross_out = cross_out.squeeze(0)    # [B,D]
        R_expand  = R.squeeze(0).repeat(stk_repr.size(0), 1)  # [B,D]

        final = torch.cat([stk_repr, R_expand, cross_out], dim=-1)  # [B,3D]

        # 8) 预测
        return self.head(final).squeeze(-1)                         # [B]