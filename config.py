# coding: utf-8
"""
配置文件
"""
import torch
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # 数据路径
    data_dir = Path("./data")
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"

    # 原始数据文件
    price_daily_file = raw_dir / "stock_price_day.parquet"
    price_30m_file = raw_dir / "stock_price_30m.parquet"
    stock_info_file = raw_dir / "stock_info.csv"
    is_suspended_file = raw_dir / "is_suspended.csv"
    is_st_file = raw_dir / "is_st_stock.csv"
    trading_day_file = raw_dir / "trading_day.csv"

    # 处理后的数据文件
    label_file = processed_dir / "weekly_labels.parquet"
    train_features_file = processed_dir / "train_features.h5"
    val_features_file = processed_dir / "val_features.h5"
    test_features_file = processed_dir / "test_features.h5"
    scaler_file = processed_dir / "scalers.pkl"
    universe_file = processed_dir / "universe.pkl"

    # 时间参数
    daily_window = 120  # 日线窗口长度
    min30_window = 80  # 30min窗口长度 (10d * 8)
    sample_freq = "W-FRI"  # 每周五采样

    # 过滤规则
    ipo_cut = 160  # 上市≤160日的新股剔除
    st_exclude = True  # 是否剔除ST
    suspended_exclude = True  # 是否剔除停牌
    min_daily_volume = 1e7  # 最小成交额过滤（元）

    # 训练参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden = 128
    batch_size = 512
    lr = 3e-4
    weight_decay = 1e-2
    alpha = 0.7  # IC Loss权重
    epochs = 20
    num_workers = 4

    # 时间划分
    train_end_year = 2021
    val_end_year = 2022

    def __post_init__(self):
        # 创建必要的目录
        self.processed_dir.mkdir(parents=True, exist_ok=True)


CFG = Config()