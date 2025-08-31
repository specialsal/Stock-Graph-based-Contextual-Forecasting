# coding: utf-8
"""
全局配置
"""
import torch
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # -------------------- 目录 --------------------
    data_dir = Path("./data")
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"

    # -------------------- 原始数据 --------------------
    price_daily_file   = raw_dir / "stock_price_day.parquet"
    price_30m_file     = raw_dir / "stock_price_30m.parquet"
    stock_info_file    = raw_dir / "stock_info.csv"
    is_suspended_file  = raw_dir / "is_suspended.csv"
    is_st_file         = raw_dir / "is_st_stock.csv"
    trading_day_file   = raw_dir / "trading_day.csv"

    # 行业 / 板块映射（两张静态表）
    industry_map_file  = raw_dir / "stock_industry_map.csv"   # 列: stock,industry
    sector_map_file    = raw_dir / "stock_sector_map.csv"     # 列: stock,sector

    # -------------------- 处理后数据 --------------------
    label_file          = processed_dir / "weekly_labels.parquet"
    train_features_file = processed_dir / "train_features.h5"
    val_features_file   = processed_dir / "val_features.h5"
    test_features_file  = processed_dir / "test_features.h5"
    scaler_file         = processed_dir / "scalers.pkl"
    universe_file       = processed_dir / "universe.pkl"

    # -------------------- 窗口长度 --------------------
    daily_window = 120     # 日线历史长度
    min30_window = 80      # 30-min 历史长度
    sample_freq  = "W-FRI" # 截面采样频率

    # -------------------- 过滤规则 --------------------
    ipo_cut           = 160
    st_exclude        = True
    suspended_exclude = True
    min_daily_volume  = 1e7

    # -------------------- 训练超参 --------------------
    device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden        = 128               # 时序编码隐藏维
    ind_emb_dim   = 32                # 行业 Embedding 维
    sec_emb_dim   = 32                # 板块 Embedding 维
    attn_heads    = 8                 # 多头注意力头数
    batch_size    = 1024               # 仅用于常规 DataLoader，不再使用（按日成批）
    lr            = 3e-4
    weight_decay  = 1e-2
    alpha         = 0.7               # IC Loss 权重
    epochs        = 200
    num_workers   = 20

    # -------------------- 数据集年份 --------------------
    train_end_year = 2021
    val_end_year   = 2022

    # -------------------- 显存/效率相关（新增） --------------------
    # 训练时每个交易日最多采样的股票数（None 表示不限制）
    max_stocks_per_day_train = 2048
    # 验证/测试时，每日先分块编码（encode）再一次性交互（interact），该参数为编码时的分块大小
    val_encode_chunk = 2048
    # 混合精度
    use_amp   = True
    amp_dtype = "bf16"   # 可选: "bf16" 或 "fp16"

    def __post_init__(self):
        self.processed_dir.mkdir(parents=True, exist_ok=True)


CFG = Config()