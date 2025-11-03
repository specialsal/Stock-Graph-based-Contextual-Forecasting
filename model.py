# ====================== model.py ======================
# coding: utf-8
"""
GCF-Net v2（两路行业图，GAT-only）
- 门控嵌入：来自 chain_sector（stock_style_map.csv 的 chain_sector）
- 图聚合：industry2 与 industry 两路 EGAT（固定 hybrid，无开关）
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
        self.w = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, 1))
    def forward(self, x):  # x:[B,T,H]
        a = torch.softmax(self.w(x), 1)  # [B,T,1]
        return (a * x).sum(1)            # [B,H]

# ----------------------------------------------------------------------
# 2. 日频特征编码器
# ----------------------------------------------------------------------
class DailyEncoder(nn.Module):
    def __init__(self, in_dim, hidden=CFG.hidden, tr_layers=CFG.tr_layers):
        super().__init__()
        self.conv = GLUConv(in_dim, hidden, k=3, p=1)
        self.tr   = RelPosTransformer(hidden, nhead=4, layers=tr_layers)
        self.pool = AttnPooling(hidden)
    def forward(self, x):  # x:[B,T,C] or [B,T,1,C]
        if x.ndim == 4 and x.shape[2] == 1:
            x = x.squeeze(2)
        x = x.transpose(1, 2)            # [B,C,T]
        x = self.conv(x)                 # [B,H,T]
        x = x.transpose(1, 2)            # [B,T,H]
        x = self.tr(x)                   # [B,T,H]
        return self.pool(x)              # [B,H]

# ----------------------------------------------------------------------
# 3. EGAT（图注意力，按分组）
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
        # 按组构建完全子图注意力（O(∑M_i^2)）
        for ind in ind_id.unique():
            idx = (ind_id == ind).nonzero(as_tuple=True)[0]
            if idx.numel() <= 1:
                out[idx] = Wh[idx]  # 退化为自身
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
# 5. 主模型（GAT-only + 双路行业图 + chain_sector 门控）
# ----------------------------------------------------------------------
class GCFNet(nn.Module):
    def __init__(
        self,
        d_in: int,
        n_chain: int,             # chain_sector 类别数（不含 pad）
        n_ind1: int,              # 行业一级数（不含 pad）
        n_ind2: int,              # 行业二级数（不含 pad）
        ctx_dim: int = CFG.ctx_dim,
        hidden: int = CFG.hidden,
        chain_emb_dim: int = CFG.chain_emb,
        tr_layers: int = CFG.tr_layers,
        gat_layers: int = CFG.gat_layers
    ):
        super().__init__()
        self.daily_enc = DailyEncoder(d_in, hidden, tr_layers=tr_layers)

        # 三套类别总数 = +1 作为 padding
        self.n_chain_total = n_chain + 1
        self.n_ind1_total  = n_ind1 + 1
        self.n_ind2_total  = n_ind2 + 1

        self.pad_chain = self.n_chain_total - 1
        self.pad_ind1  = self.n_ind1_total  - 1
        self.pad_ind2  = self.n_ind2_total  - 1

        # chain_sector 嵌入 -> 门控融合用
        self.chain_emb  = nn.Embedding(self.n_chain_total, chain_emb_dim, padding_idx=self.pad_chain)
        self.chain_proj = nn.Linear(chain_emb_dim, hidden)

        # 门控：h_d vs h_chain
        self.gate = nn.Linear(hidden * 2, 1)

        # 上下文归一
        self.ctx_norm = nn.LayerNorm(ctx_dim)
        self.film = FiLM(ctx_dim, hidden)

        # 两路 GAT 图块（固定 hybrid）
        self.graph_ind2 = DynamicGraphBlock(hidden, layers=gat_layers)
        self.graph_ind1 = DynamicGraphBlock(hidden, layers=gat_layers)
        self.fuse_12 = nn.Linear(hidden * 2, hidden)   # 融合两路图
        self.fuse_norm = nn.LayerNorm(hidden)

        # 输出头
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 3),
            nn.Linear(hidden * 3, 192), nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(192, 64), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self,
                daily_feat: torch.Tensor,
                chain_id: torch.Tensor,
                ind1_id: torch.Tensor,
                ind2_id: torch.Tensor,
                ctx_feat: torch.Tensor) -> torch.Tensor:
        # 日频编码
        h_d = self.daily_enc(daily_feat)                  # [N,H]

        # chain_sector 嵌入用于门控
        h_chain = self.chain_proj(self.chain_emb(chain_id.clamp_max(self.pad_chain)))  # [N,H]
        g = torch.sigmoid(self.gate(torch.cat([h_d, h_chain], -1))).squeeze(-1)
        h_price = g.unsqueeze(-1) * h_d + (1 - g.unsqueeze(-1)) * h_chain              # [N,H]

        # FiLM with context
        ctx_n = self.ctx_norm(ctx_feat)
        h_ctx = self.film(h_price, ctx_n)                 # [N,H]

        # 两路行业图（GAT-only）
        ind1_id = ind1_id.clamp_max(self.pad_ind1)
        ind2_id = ind2_id.clamp_max(self.pad_ind2)
        h_g2 = self.graph_ind2(h_ctx, ind2_id)            # [N,H]
        h_g1 = self.graph_ind1(h_ctx, ind1_id)            # [N,H]

        # 门控融合两路图
        g12 = torch.sigmoid(self.fuse_12(torch.cat([h_g2, h_g1], -1)))  # [N,H]
        h_graph = self.fuse_norm(g12 * h_g2 + (1 - g12) * h_g1)         # [N,H]

        out = torch.cat([h_price, h_ctx, h_graph], -1)    # [N,3H]
        return self.head(out).squeeze(-1)                 # [N]