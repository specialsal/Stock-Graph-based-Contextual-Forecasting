# coding: utf-8
"""
全局配置（优化版）
"""
import torch
from dataclasses import dataclass
from pathlib import Path
import random, numpy as np


@dataclass
class Config:
    # ---------------- 目录 ----------------
    data_dir = Path("./data")
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"

    # ---------------- 原始数据 ----------------
    price_daily_file = raw_dir / "stock_price_day.parquet"
    price_30m_file = raw_dir / "stock_price_30m.parquet"
    stock_info_file = raw_dir / "stock_info.csv"
    is_suspended_file = raw_dir / "is_suspended.csv"
    is_st_file = raw_dir / "is_st_stock.csv"
    trading_day_file = raw_dir / "trading_day.csv"

    industry_map_file = raw_dir / "stock_industry_map.csv"
    sector_map_file = raw_dir / "stock_sector_map.csv"

    # ---------------- 处理后数据 ----------------
    label_file = processed_dir / "weekly_labels.parquet"
    train_features_file = processed_dir / "train_features.h5"
    val_features_file = processed_dir / "val_features.h5"
    test_features_file = processed_dir / "test_features.h5"
    scaler_file = processed_dir / "scalers.pkl"
    universe_file = processed_dir / "universe.pkl"

    # ---------------- 数据集划分 ----------------
    train_end_year = 2022
    val_end_year = 2023

    # ---------------- 窗口长度 ----------------
    daily_window = 120
    min30_window = 80
    sample_freq = "W-FRI"

    # ---------------- 过滤规则 ----------------
    ipo_cut = 160
    st_exclude = True
    suspended_exclude = True
    min_daily_volume = 1e7

    # ---------------- 训练超参 ----------------
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden = 128
    ind_emb_dim = 32
    sec_emb_dim = 32
    attn_heads = 8
    lr = 3e-4
    weight_decay = 1e-2
    alpha = 0.7
    epochs = 200

    # ---------------- 优化设置 ----------------
    num_workers = 20  # 降低worker数量，避免过度竞争
    max_stocks_per_day_train = None
    val_encode_chunk = 2048
    use_amp = True
    amp_dtype = "bf16"  # 使用bfloat16获得更好的数值稳定性

    def __post_init__(self):
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # 全局随机种子
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)

        # CUDA优化设置
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False  # 为了性能牺牲一点确定性


CFG = Config()