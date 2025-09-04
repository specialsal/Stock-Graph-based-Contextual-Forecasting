# ====================== config.py ======================
# coding: utf-8
"""
全局配置 —— 提速版（含图模块简化、AMP/Loader优化、指标与模型选择）
本版本说明：
- 滚动训练阶段不再读取 CFG.scaler_file，不再使用“全局 Scaler”
- 每个训练窗口会在线拟合当期训练集 Scaler，避免未来信息泄漏
- 新增 ranking_weight: 训练损失 = w*(1 - Pearson) + (1-w)*PairwiseRanking
"""
import torch, random, numpy as np
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    # -------- 路径 --------
    data_dir      = Path("./data")
    raw_dir       = data_dir / "raw"
    processed_dir = data_dir / "processed"
    feat_file     = processed_dir / "features_daily.h5"
    scaler_file   = processed_dir / "scaler.pkl"  # 兼容旧字段，但训练阶段不会读取/使用
    label_file    = processed_dir / "weekly_labels.parquet"
    universe_file = processed_dir / "universe.pkl"
    model_dir     = processed_dir / "models"
    registry_file = processed_dir / "models" / "model_registry.csv"

    # -------- 原始数据文件 --------
    price_day_file     = raw_dir / "stock_price_day.parquet"
    index_day_file     = raw_dir / "index_price_day.parquet"
    style_day_file     = raw_dir / "sector_price_day.parquet"
    trading_day_file   = raw_dir / "trading_day.csv"
    industry_map_file  = raw_dir / "stock_industry_map.csv"
    stock_info_file     = raw_dir / "stock_info.csv"
    is_suspended_file   = raw_dir / "is_suspended.csv"
    is_st_file          = raw_dir / "is_st_stock.csv"

    # -------- 股票筛选（可选） --------
    enable_filters      = True       # 是否启用筛选（关闭则与旧逻辑一致）
    ipo_cut_days        = 120        # 新股上市满多少天才纳入（自然日）
    suspended_exclude   = True       # 排除停牌
    st_exclude          = True       # 排除 ST
    min_daily_turnover  = 5e6        # 当日最低成交额（单位与原始数据一致，如人民币元）
    allow_missing_info  = False      # 缺少基础信息的股票是否保留（默认丢弃）

    # -------- 特征窗口 --------
    daily_window = 20

    # -------- 滚动窗参数 --------
    train_years = 5
    val_weeks   = 52
    step_weeks  = 52
    start_date  = "2011-01-01"
    end_date    = "2025-08-26"

    # -------- 模型超参 --------
    hidden        = 64
    ind_emb       = 16
    ctx_dim       = 21
    tr_layers     = 1
    gat_layers    = 1
    graph_type    = "gat"        # "mean" 或 "gat"
    topk_per_ind  = 16
    lr            = 3e-4
    weight_decay  = 1e-2
    epochs_warm   = 10

    # -------- 训练细节 --------
    batch_size        = 3500
    grad_accum_steps  = 4
    device            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers       = 8
    prefetch_factor   = 4
    persistent_workers= True
    pin_memory        = True
    use_amp           = True
    amp_dtype         = "bf16"           # "bf16" 或 "fp16"
    use_tf32          = True
    use_torch_compile = False
    print_step_interval = 10

    # -------- 选择与记录 --------
    select_metric   = "rankic"  # 基于验证集平均 RankIC 选择
    score_alpha     = 0.5       # 全局最优时的风险调整：score = mean - alpha * std
    recent_topN     = 5         # 最近 N 个窗口里挑一个最优（输出 best_recent_N.pth）

    # -------- 损失加权（新增） --------
    ranking_weight  = 0.5           # loss = w*(1 - Pearson) + (1-w)*pairwise_rank

    # -------- 其他 --------
    seed = 42

    def __post_init__(self):
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)

CFG = Config()