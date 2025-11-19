# ====================== config.py ======================
# coding: utf-8
"""
全局配置 —— RankNet_margin(cost=m) + 精简日志 + 消融开关版本

修改点：
- 保留你原有的训练/数据/超参配置；
- 新增三类“消融开关”，用于控制模型中的三个模块是否启用：
  1) graph_type 支持 "none"（移除 GNN/行业图）
  2) use_film: False 时移除 FiLM（上下文调制）
  3) use_transformer: False 时移除 Transformer（改为 GLUConv + AttnPooling）

使用建议（为避免不同实验互相覆盖，建议为每次实验设置独立 run_name）：
- 基线：run_name = "TGF-model"                （graph_type="gat"/"mean"，use_film=True，use_transformer=True）
- No-GNN：run_name = "TGF-abl_noGNN"          （graph_type="none"）
- No-FiLM：run_name = "TGF-abl_noFiLM"        （use_film=False）
- No-TR：run_name = "TGF-abl_noTR"            （use_transformer=False）

注意：当更改 run_name 时，会自动将模型与回测输出保存到独立目录，互不干扰。
"""
import torch, random, numpy as np
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

@dataclass
class Config:
    # -------- 运行命名（影响模型输出路径与回测路径）--------
    # 建议每次实验改一个 run_name，避免覆盖
    # run_name = "TGF-model"          # 基线
    # run_name = "TGF-abl_noGNN"      # 移除 GNN（graph_type="none"）
    # run_name = "TGF-abl_noFiLM"     # 移除 FiLM（use_film=False）
    # run_name = "TGF-abl_noTR"       # 移除 Transformer（use_transformer=False）
    # run_name = "TGF-model_5_26_26"
    run_name = "TGF-fundamental"

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
    price_fundamental_file     = raw_dir / "stock_fundamental_day.parquet"
    index_day_file     = raw_dir / "index_price_day.parquet"
    style_day_file     = raw_dir / "sector_price_day.parquet"
    trading_day_file   = raw_dir / "trading_day.csv"
    industry_map_file  = raw_dir / "stock_industry_map.csv"
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

    # -------- 模型超参（与消融相关的开关在此处） --------
    hidden        = 64
    ind_emb       = 16
    ctx_dim       = 21
    tr_layers     = 1
    gat_layers    = 1

    # 行业图类型：
    #   "gat"  -> 动态注意力行业图（慢）
    #   "none" -> 移除 GNN（用于消融：仅 Transformer + FiLM）
    graph_type    = "gat"

    # 消融开关：
    # - use_film=False     -> 移除 FiLM（上下文调制），等价 h_ctx=h_price
    # - use_transformer=False -> 移除 Transformer，日频编码为 GLUConv + AttnPooling
    use_film         = True
    use_transformer  = True

    lr            = 1e-4
    weight_decay  = 1e-2

    # -------- 训练细节 --------
    batch_size        = 4098
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
    pair_margin_m  = 0           # 常数 margin（如 0.0025 表示约 25 bps；你目前为 0）
    pair_num_pairs = 4096        # 每步随机采样的成对数量

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
        # 目录就绪与随机种子
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        (Path("./models") / "tblogs").mkdir(parents=True, exist_ok=True)  # 统一 TensorBoard 目录
        import random as _random
        _random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)

CFG = Config()