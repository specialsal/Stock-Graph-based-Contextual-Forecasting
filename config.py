# ====================== config.py ======================
# coding: utf-8
"""
全局配置 —— RankNet_margin(cost=m) + 精简日志版本
本版变更：
- 移除 graph_type 与所有 mean 图相关开关，仅保留 GAT。
- 新增 style_map_file（chain_sector 嵌入来源）、chain_emb 维度。
- 行业图固定为 hybrid：industry2 与 industry 两路 GAT 聚合并融合（无开关）。
"""
import torch, random, numpy as np
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

@dataclass
class Config:
    # -------- 运行命名（影响模型输出路径）--------
    run_name = "tr1gat1win50_margin_m0_hybrid2graph"

    # -------- 路径 --------
    data_dir      = Path("./data")
    raw_dir       = data_dir / "raw"
    processed_dir = data_dir / "processed"
    model_dir     = Path(f"./models/model_{run_name}")     # 按 run_name 组织
    feat_file     = processed_dir / "features_daily.h5"
    label_file    = processed_dir / "weekly_labels.parquet"
    registry_file = model_dir / "model_registry.csv"       # 窗口登记表

    # -------- 原始数据文件 --------
    price_day_file     = raw_dir / "stock_price_day.parquet"
    index_day_file     = raw_dir / "index_price_day.parquet"
    style_day_file     = raw_dir / "sector_price_day.parquet"
    trading_day_file   = raw_dir / "trading_day.csv"
    industry_map_file  = raw_dir / "stock_industry_map.csv"   # 含 industry, industry2
    style_map_file     = raw_dir / "stock_style_map.csv"      # 含 sector, chain_sector
    stock_info_file     = raw_dir / "stock_info.csv"
    is_suspended_file   = raw_dir / "is_suspended.csv"
    is_st_file          = raw_dir / "is_st_stock.csv"

    # -------- RAM 加速（窗口级一次性常驻内存）--------
    ram_accel_enable     = True
    ram_accel_mem_cap_gb = 48

    # -------- 股票筛选（可选） --------
    enable_filters      = True
    ipo_cut_days        = 120
    suspended_exclude   = True
    st_exclude          = True
    min_daily_turnover  = 5e6
    allow_missing_info  = False

    # -------- 特征窗口 --------
    max_lookback = 120
    daily_window = 50

    # -------- 滚动窗参数（单位=周）--------
    train_years = 5
    val_weeks   = 52
    step_weeks  = 52
    start_date  = "2011-01-01"
    end_date    = datetime.today().strftime("%Y-%m-%d")

    # -------- 模型超参 --------
    hidden        = 64
    ind_emb       = 16         # 若不区分，可与 chain_emb 相同
    chain_emb     = 16         # chain_sector 嵌入维度
    ctx_dim       = 21
    tr_layers     = 1
    gat_layers    = 1          # 每路 GAT 层数（industry 与 industry2 各用同样层数）
    lr            = 1e-4
    weight_decay  = 1e-2

    # -------- 训练细节 --------
    batch_size        = 8196
    grad_accum_steps  = 1
    device            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers       = 8
    prefetch_factor   = 4
    persistent_workers= True
    pin_memory        = True
    use_amp           = True
    amp_dtype         = "bf16"   # "bf16" 或 "fp16"
    use_tf32          = True
    use_torch_compile = False
    print_step_interval = 10

    # -------- Pairwise RankNet (margin) 参数 --------
    pair_margin_m  = 0       # 常数 margin（可设 0, 0.0025 等）
    pair_num_pairs = 4096    # 每步随机采样的成对数量

    # -------- 早停参数（以验证集 ic_rank 为唯一判据）--------
    early_stop_min_epochs  = 3
    early_stop_max_epochs  = 15
    early_stop_patience    = 3
    early_stop_min_delta   = 1e-4

    # -------- Warm Start 开关 --------
    warm_start_enable = False
    warm_start_strict = False  # 推荐 False，允许部分加载

    # -------- 板块开关 --------
    include_star_market = False
    include_chinext = False
    include_bse = False
    include_neeq = False

    # -------- 其他 --------
    seed = 42

    def __post_init__(self):
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        (Path("./models") / "tblogs").mkdir(parents=True, exist_ok=True)  # 统一 TensorBoard 目录
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)

CFG = Config()