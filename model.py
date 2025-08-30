# coding: utf-8
"""
模型定义
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import CFG


class DailyEncoder(nn.Module):
    """日频Transformer编码器"""

    def __init__(self, in_dim=6, hidden=128):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden)
        self.pe = nn.Parameter(torch.randn(1, CFG.daily_window, hidden) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=4,
            dim_feedforward=hidden * 4,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):
        # x: [B, T, C]
        x = self.input_proj(x)
        x = x + self.pe
        x = self.encoder(x)
        x = self.norm(x)
        # Global average pooling
        return x.mean(dim=1)


class MinuteEncoder(nn.Module):
    """30分钟Conv + GRU编码器"""

    def __init__(self, in_dim=6, hidden=128):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, hidden // 2, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm1d(hidden // 2)
        self.conv2 = nn.Conv1d(hidden // 2, hidden, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden)

        self.gru = nn.GRU(
            hidden,
            hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        # x: [B, T, C] -> Conv需要 [B, C, T]
        x = x.transpose(1, 2)

        # CNN特征提取
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # [B, C, T] -> [B, T, C]
        x = x.transpose(1, 2)

        # GRU序列建模
        out, _ = self.gru(x)

        # Attention聚合
        weights = torch.softmax(self.attention(out), dim=1)
        output = (out * weights).sum(dim=1)

        return output


class FusionModel(nn.Module):
    """多模态融合模型"""

    def __init__(self):
        super().__init__()

        # 编码器
        self.daily_encoder = DailyEncoder(in_dim=6, hidden=CFG.hidden)
        self.minute_encoder = MinuteEncoder(in_dim=6, hidden=CFG.hidden)

        # 跨模态交互
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=CFG.hidden,
            num_heads=4,
            batch_first=True
        )

        # 融合层
        fusion_dim = CFG.hidden + CFG.hidden * 2  # daily + minute(bidirectional)

        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, daily, min30):
        # 特征编码
        h_daily = self.daily_encoder(daily)  # [B, hidden]
        h_minute = self.minute_encoder(min30)  # [B, hidden*2]

        # 跨模态注意力
        h_daily_expanded = h_daily.unsqueeze(1)  # [B, 1, hidden]
        h_minute_part = h_minute[:, :CFG.hidden].unsqueeze(1)  # [B, 1, hidden]

        cross_out, _ = self.cross_attention(
            h_daily_expanded,
            h_minute_part,
            h_minute_part
        )
        cross_out = cross_out.squeeze(1)  # [B, hidden]

        # 特征融合
        h_combined = torch.cat([cross_out, h_minute], dim=-1)

        # 预测
        output = self.fusion(h_combined)
        return output.squeeze(-1)