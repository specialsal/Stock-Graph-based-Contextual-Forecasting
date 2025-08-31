# coding: utf-8
"""
模型定义：多尺度时序编码 + 行业/板块 Embedding + 截面交互（O(B) 注意力）
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
    * 截面交互          = 市场隐状态 R（Q=cls, K/V=stocks）→ 截面 B → 反馈到个股
      注意力复杂度：全部为 O(B)，避免 O(B^2) 显存爆炸
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
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.stk_dim) * 0.02)

        # 双阶段多头注意力（batch_first: [B, L, D]）
        self.attn1 = nn.MultiheadAttention(self.stk_dim, num_heads=CFG.attn_heads, batch_first=True, dropout=0.1)
        self.attn2 = nn.MultiheadAttention(self.stk_dim, num_heads=CFG.attn_heads, batch_first=True, dropout=0.1)

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

    # ---------- 将原三路输入编码为单只股票表征 ----------
    def encode(self,
               daily:  torch.Tensor,                                # [B,Td,6]
               minute: torch.Tensor,                                # [B,Tm,6]
               ind_id: torch.Tensor,                                # [B]   int
               sec_id: torch.Tensor) -> torch.Tensor:               # return [B, D]
        h_d  = self.daily_enc(daily)                                # [B,128]
        h_m  = self.minute_enc(minute)                              # [B,256]
        h_ind = self.ind_emb(ind_id)                                # [B,32]
        h_sec = self.sec_emb(sec_id)                                # [B,32]
        stk_repr = torch.cat([h_d, h_m, h_ind, h_sec], dim=-1)      # [B,D]
        stk_repr = self.ln_in(stk_repr)
        return stk_repr                                             # [B,D]

    # ---------- 截面交互 + 头部 ----------
    def interact_and_head(self, stk_repr: torch.Tensor) -> torch.Tensor:
        """
        输入: stk_repr [B,D]
        过程:
          1) R = Attn(cls -> stocks)              O(B)
          2) B = Attn(R -> stocks)                O(B)
          3) cross = Attn(stocks -> B)            O(B)（K/V 长度为 1）
          4) 拼接 [stk | R | cross] -> head
        输出: [B]
        """
        Bsz = stk_repr.size(0)

        # 视作序列：[1, B, D]
        stk_seq = stk_repr.unsqueeze(0)

        # 1) 市场隐状态 R：Q=cls_token, K/V=stocks
        R, _ = self.attn1(self.cls_token, stk_seq, stk_seq)         # [1,1,D]

        # 2) 第一次交叉注意力：R → Stocks，生成 B（市场基底）
        B_vec, _ = self.attn1(R, stk_seq, stk_seq)                  # [1,1,D]

        # 3) 第二次交叉注意力：Stocks → B（不再 repeat 到 B）
        cross_out, _ = self.attn2(stk_seq, B_vec, B_vec)            # [1,B,D]

        # 4) 拼接三个向量
        stk_seq   = stk_seq.squeeze(0)                               # [B,D]
        cross_out = cross_out.squeeze(0)                             # [B,D]
        R_expand  = R.squeeze(0).repeat(Bsz, 1)                      # [B,D]

        final = torch.cat([stk_seq, R_expand, cross_out], dim=-1)    # [B,3D]
        return self.head(final).squeeze(-1)                          # [B]

    # ---------- 标准前向 ----------
    def forward(self,
                daily:  torch.Tensor,                                # [B,Td,6]
                minute: torch.Tensor,                                # [B,Tm,6]
                ind_id: torch.Tensor,                                # [B]   int
                sec_id: torch.Tensor):                               # [B]   int
        stk_repr = self.encode(daily, minute, ind_id, sec_id)        # [B,D]
        return self.interact_and_head(stk_repr)                      # [B]