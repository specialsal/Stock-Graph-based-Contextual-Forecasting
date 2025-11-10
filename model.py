# ====================== model.py ======================
# coding: utf-8
"""
GCF-Net v2（稀疏行业图版本）- 提速版 + 消融开关
新增三类消融控制（最小改动保证 head 维度恒定）：
- 移除 GNN：graph_type="none" -> 行业图块替换为 Identity，h_graph := h_ctx（保持 [B,H]）
- 移除 FiLM：use_film=False -> 直接 h_ctx := h_price（忽略上下文调制）
- 移除 Transformer：use_transformer=False -> DailyEncoder 退化为 GLUConv + AttnPooling（不使用自注意力）

说明：
- 为减少对训练/回测流程的影响，head 的输入维度固定为 3H（concat[h_price, h_ctx, h_graph]）。
- 当关闭某一模块时，用语义合理的向量占位以保持维度：
  - No-GNN: h_graph = h_ctx
  - No-FiLM: h_ctx = h_price
  - No-TR : DailyEncoder 仍输出 H 维（由轻量替代产生）
"""
import torch, torch.nn as nn, torch.nn.functional as F
from config import CFG

# ----------------------------------------------------------------------
# 1. 基础组件
# ----------------------------------------------------------------------
class GLUConv(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv1d(in_c, out_c * 2, k, s, p)

    def forward(self, x):  # x:[B,C,T]
        a, b = self.conv(x).chunk(2, dim=1)
        return a * torch.sigmoid(b)

class RelPosTransformer(nn.Module):
    def __init__(self, dim, nhead=4, layers=1, p_drop=0.1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            dim, nhead=nhead, dim_feedforward=dim * 4,
            dropout=p_drop, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, layers)

    def forward(self, x):  # x:[B,T,H]
        return self.encoder(x)

class AttnPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Sequential(
            nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, 1)
        )

    def forward(self, x):  # x:[B,T,H]
        a = torch.softmax(self.w(x), 1)  # [B,T,1]
        return (a * x).sum(1)            # [B,H]

# ----------------------------------------------------------------------
# 2. 日频特征编码器（支持移除 Transformer）
# ----------------------------------------------------------------------
class DailyEncoder(nn.Module):
    """
    - 正常：GLUConv -> Transformer -> AttnPooling
    - 移除 Transformer：use_transformer=False -> GLUConv -> AttnPooling（不使用自注意力）
    """
    def __init__(self, in_dim, hidden=CFG.hidden, tr_layers=CFG.tr_layers, use_transformer: bool = True):
        super().__init__()
        self.use_transformer = bool(use_transformer)
        self.conv = GLUConv(in_dim, hidden, k=3, p=1)
        if self.use_transformer:
            self.tr = RelPosTransformer(hidden, nhead=4, layers=tr_layers)
        else:
            # 不使用 Transformer 时，不定义 self.tr，直接用池化
            self.tr = None
        self.pool = AttnPooling(hidden)

    def forward(self, x):  # x:[B,T,C] or [B,T,1,C]
        # 统一形状：将 [B,T,1,C] 压成 [B,T,C]
        if x.ndim == 4 and x.shape[2] == 1:
            x = x.squeeze(2)
        # [B,T,C] -> [B,C,T]
        x = x.transpose(1, 2)
        # GLUConv：[B,C,T] -> [B,H,T]
        x = self.conv(x)
        # -> [B,T,H]
        x = x.transpose(1, 2)
        # Transformer（可选）
        if self.use_transformer and (self.tr is not None):
            x = self.tr(x)  # [B,T,H]
        # 注意力池化 -> [B,H]
        return self.pool(x)

# ----------------------------------------------------------------------
# 3A. 提速版行业“均值图”块（O(N)）
# ----------------------------------------------------------------------
class IndustryMeanBlock(nn.Module):
    def __init__(self, hidden=CFG.hidden, layers=CFG.gat_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([nn.Identity() for _ in range(layers)])
        self.norms  = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(hidden * 2, hidden)

    @staticmethod
    def group_mean(h, ind_id):
        # h:[N,H], ind_id:[N], 计算每个行业均值，并按样本顺序聚合回去
        N, H = h.shape
        K = int(ind_id.max().item()) + 1
        sum_buf = torch.zeros(K, H, device=h.device, dtype=h.dtype)
        cnt_buf = torch.zeros(K, 1, device=h.device, dtype=h.dtype)
        sum_buf.index_add_(0, ind_id, h)
        ones = torch.ones(N, 1, device=h.device, dtype=h.dtype)
        cnt_buf.index_add_(0, ind_id, ones)
        mean = sum_buf / (cnt_buf + 1e-6)   # [K,H]
        return mean[ind_id]                 # [N,H]

    def forward(self, h, ind_id):
        # 简单 residual + gating 到行业均值
        for layer, norm in zip(self.layers, self.norms):
            h_in = layer(h)
            ind_mean = self.group_mean(h_in, ind_id)
            g = torch.sigmoid(self.gate(torch.cat([h_in, ind_mean], -1)))
            h = norm(h_in + self.dropout(g * ind_mean))
        return h

# ----------------------------------------------------------------------
# 3B. 原 EGAT（保留以备切换）
# ----------------------------------------------------------------------
class EGATLayer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.W   = nn.Linear(hidden, hidden, bias=False)
        self.att = nn.Linear(hidden * 2 + 1, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, h, ind_id):  # h:[N,H]  ind_id:[N]
        Wh = self.W(h)             # [N,H]
        out = torch.zeros_like(Wh)
        # 注意：该实现 O(∑M_i^2) 很慢，仅在 graph_type="gat" 时使用
        for ind in ind_id.unique():
            idx = (ind_id == ind).nonzero(as_tuple=True)[0]
            if idx.numel() <= 1:
                continue
            Wh_sub = Wh[idx]       # [M,H]
            M = len(idx)
            Wh_i = Wh_sub.unsqueeze(1).expand(-1, M, -1)
            Wh_j = Wh_sub.unsqueeze(0).expand(M, -1, -1)
            e_ij = torch.ones(M, M, 1, device=h.device)
            e = self.leaky_relu(self.att(torch.cat([Wh_i, Wh_j, e_ij], dim=-1))).squeeze(-1)  # [M,M]
            alpha = torch.softmax(e, 1)
            out[idx] = alpha @ Wh_sub
        return out

class DynamicGraphBlock(nn.Module):
    def __init__(self, hidden=CFG.hidden, layers=CFG.gat_layers):
        super().__init__()
        self.layers = nn.ModuleList([EGATLayer(hidden) for _ in range(layers)])
        self.norms  = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])

    def forward(self, h, ind_id):
        for gat, norm in zip(self.layers, self.norms):
            h = norm(gat(h, ind_id) + h)
        return h

# ----------------------------------------------------------------------
# 4. FiLM
# ----------------------------------------------------------------------
class FiLM(nn.Module):
    def __init__(self, ctx_dim, hidden):
        super().__init__()
        self.gamma = nn.Linear(ctx_dim, hidden)
        self.beta  = nn.Linear(ctx_dim, hidden)
    def forward(self, h, ctx):
        return h * (1 + self.gamma(ctx)) + self.beta(ctx)

# ----------------------------------------------------------------------
# 5. 主模型（含消融开关）
# ----------------------------------------------------------------------
class GCFNet(nn.Module):
    def __init__(
        self,
        d_in: int,
        n_ind: int,                 # 已知行业类别数（不含未知）
        ctx_dim: int = CFG.ctx_dim,
        hidden: int = CFG.hidden,
        ind_emb_dim: int = CFG.ind_emb,
        graph_type: str = CFG.graph_type,   # 新增支持 "none"
        tr_layers: int = CFG.tr_layers,
        gat_layers: int = CFG.gat_layers,
        use_film: bool = True,             # 新增：移除 FiLM
        use_transformer: bool = True       # 新增：移除 Transformer
    ):
        """
        参数说明（与消融相关）：
        - graph_type:
            "gat"  -> 动态注意力行业图（慢）
            其他   -> 行业均值图（快）
            "none" -> 移除行业图，forward 中用 h_graph := h_ctx
        - use_film: False -> 移除 FiLM（h_ctx := h_price）
        - use_transformer: False -> DailyEncoder 无 Transformer（仅 GLUConv + AttnPooling）
        """
        super().__init__()
        # 日频编码器（含/不含 Transformer）
        self.daily_enc = DailyEncoder(d_in, hidden, tr_layers=tr_layers, use_transformer=use_transformer)

        # 行业嵌入
        self.n_ind_total = n_ind + 1
        self.pad_index   = self.n_ind_total - 1
        self.ind_emb  = nn.Embedding(self.n_ind_total, ind_emb_dim, padding_idx=self.pad_index)
        self.ind_proj = nn.Linear(ind_emb_dim, hidden)

        # 价格与行业门控
        self.gate = nn.Linear(hidden * 2, 1)

        # FiLM 与上下文归一化（可关闭）
        self.use_film = bool(use_film)
        self.ctx_dim = int(ctx_dim)
        if self.use_film:
            # 对上下文做 LayerNorm（仅基于当前样本，无未来信息）
            self.ctx_norm = nn.LayerNorm(ctx_dim)
            self.film = FiLM(ctx_dim, hidden)
        else:
            # 占位，便于 forward 里少分支
            self.ctx_norm = None
            self.film = None

        # 行业图类型（可关闭）
        self.graph_type = str(graph_type).lower() if graph_type is not None else "mean"
        if self.graph_type == "gat":
            self.graph_blk = DynamicGraphBlock(hidden, layers=gat_layers)
        elif self.graph_type == "none":
            # 不使用任何行业图（用 Identity 占位，实际前向中会覆写 h_graph）
            self.graph_blk = nn.Identity()
        else:
            self.graph_blk = IndustryMeanBlock(hidden, layers=gat_layers)

        # 预测头（维度固定为 3H，方便消融时少改动）
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 3),
            nn.Linear(hidden * 3, 192), nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(192, 64), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, daily_feat: torch.Tensor, ind_id: torch.Tensor, ctx_feat: torch.Tensor) -> torch.Tensor:
        """
        输入：
        - daily_feat: [N,T,C] 或 [N,T,1,C]
        - ind_id:     [N]
        - ctx_feat:   [N,C_ctx]
        """
        # 1) 日频编码 -> [N,H]
        h_d   = self.daily_enc(daily_feat)             # [N,H]
        # 行业嵌入 -> [N,H]
        h_ind = self.ind_proj(self.ind_emb(ind_id))    # [N,H]

        # 2) 门控融合价格与行业
        g = torch.sigmoid(self.gate(torch.cat([h_d, h_ind], -1))).squeeze(-1)  # [N]
        h_price = g.unsqueeze(-1) * h_d + (1 - g.unsqueeze(-1)) * h_ind        # [N,H]

        # 3) FiLM（可选）
        if self.use_film:
            # 对 ctx 做 LayerNorm，再送入 FiLM
            ctx_n = self.ctx_norm(ctx_feat)                # [N,C_ctx]
            h_ctx = self.film(h_price, ctx_n)              # [N,H]
        else:
            # 移除 FiLM：直接绕过，保持维度
            h_ctx = h_price

        # 4) 行业图（可关闭）
        if self.graph_type == "none":
            # 移除 GNN：用 h_ctx 作为“行业图输出”的占位，维度保持 [N,H]
            h_graph = h_ctx
        else:
            h_graph = self.graph_blk(h_ctx, ind_id)        # [N,H]

        # 5) 预测头（固定 3H）
        out = torch.cat([h_price, h_ctx, h_graph], -1)     # [N,3H]
        return self.head(out).squeeze(-1)                  # [N]