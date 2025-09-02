# ====================== config.py ======================
# coding: utf-8
"""
全局配置 —— 提速版（含图模块简化、AMP/Loader优化、指标与模型选择）
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
    scaler_file   = processed_dir / "scaler.pkl"
    label_file    = processed_dir / "weekly_labels.parquet"
    universe_file = processed_dir / "universe.pkl"
    model_dir     = processed_dir / "models"
    registry_file = processed_dir / "models" / "model_registry.csv"  # 新增：记录各窗口验证表现

    # -------- 原始数据文件 --------
    price_day_file     = raw_dir / "stock_price_day.parquet"
    price_30m_file     = raw_dir / "stock_price_30m.parquet"
    index_day_file     = raw_dir / "index_price_day.parquet"
    style_day_file     = raw_dir / "sector_price_day.parquet"
    trading_day_file   = raw_dir / "trading_day.csv"
    industry_map_file  = raw_dir / "stock_industry_map.csv"

    # -------- 特征窗口 --------
    daily_window = 20

    # -------- 滚动窗参数 --------
    train_years = 5
    val_weeks   = 52
    step_weeks  = 52
    start_date  = "2011-01-01"
    end_date    = "2025-08-26"

    # -------- 模型超参（适度减小） --------
    hidden        = 64           # 原 128 -> 96
    ind_emb       = 16           # 原 32  -> 16
    ctx_dim       = 21           # 会在训练脚本里由文件实际列数覆盖
    tr_layers     = 1            # Transformer 层数从 2 -> 1
    gat_layers    = 1            # GAT/MeanGraph 层数 1 即可
    graph_type    = "mean"       # "mean"（默认, O(N)）或 "gat"
    topk_per_ind  = 16           # 若自定义其它图近邻采样可用
    lr            = 3e-4
    weight_decay  = 1e-2
    epochs_warm   = 6            # 总训练轮次；原 10，建议 6~10 之间
    # epochs_finet = 20          # 不再使用，可保留以兼容

    # -------- 训练细节 --------
    batch_size        = 1024
    grad_accum_steps  = 4                # 原 8 -> 4，减小等待时延
    device            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers       = 8                # 原 16 -> 8，缓解 CPU 与 HDF5 压力
    prefetch_factor   = 4                # DataLoader 预取
    persistent_workers= True             # 持久化 worker
    pin_memory        = True
    use_amp           = True
    amp_dtype         = "bf16"           # "bf16" 或 "fp16"
    use_tf32          = True             # Ampere+ 上建议 True
    use_torch_compile = False            # PyTorch 2.x 可尝试 True
    print_step_interval = 10             # 每多少个 step 刷新一次 tqdm 指标

    # -------- 选择与记录 --------
    select_metric   = "rankic"           # 基于验证集平均 RankIC 选择
    score_alpha     = 0.5                # 全局最优时的风险调整：score = mean - alpha * std
    recent_topN     = 5                  # 最近 N 个窗口里挑一个最优（输出 best_recent_N.pth）

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