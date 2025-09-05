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
from datetime import datetime

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

    # -------- RAM 加速（窗口级一次性常驻内存）--------
    ram_accel_enable    = True     # 默认关闭；内存大的机器可手动改为 True
    ram_accel_mem_cap_gb= 24        # RAM 加速单窗口的内存上限（GB）；超过将自动回退为逐组加载

    # -------- 股票筛选（可选） --------
    enable_filters      = True       # 是否启用筛选（关闭则与旧逻辑一致）
    ipo_cut_days        = 120        # 新股上市满多少天才纳入（自然日）
    suspended_exclude   = True       # 排除停牌
    st_exclude          = True       # 排除 ST
    min_daily_turnover  = 5e6        # 当日最低成交额（单位与原始数据一致，如人民币元）
    allow_missing_info  = False      # 缺少基础信息的股票是否保留（默认丢弃）

    # -------- 特征窗口 --------
    max_lookback = 120  # 最大回溯天数（确保足够覆盖所有日频特征）在增量更新时使用
    daily_window = 20  # 日频特征的时间窗口大小

    # -------- 滚动窗参数 --------
    train_years = 5
    val_weeks   = 52
    step_weeks  = 52
    start_date  = "2011-01-01"
    end_date    = datetime.today().strftime("%Y-%m-%d")  # 默认到今天为止

    # -------- 模型超参 --------
    hidden        = 64  # 隐藏层维度
    ind_emb       = 16  # 行业嵌入维度
    ctx_dim       = 21  # 上下文特征维度
    tr_layers     = 1   # Transformer 层数
    gat_layers    = 1  # GAT 层数
    graph_type    = "gat"        # 图模块类型"mean" 或 "gat"
    topk_per_ind  = 16  # 每个行业选取的TopK股票数量
    lr            = 3e-4    # 学习率
    weight_decay  = 1e-2    # 权重衰减
    epochs_warm   = 10  # 每个窗口训练轮数

    # -------- 训练细节 --------
    batch_size        = 3500
    grad_accum_steps  = 4 # 梯度累积步数
    device            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers       = 8   # 数据加载进程数
    prefetch_factor   = 4   # 每个进程预取批次数
    persistent_workers= True    # 是否保持数据加载进程常驻
    pin_memory        = True    # 是否固定内存（加速GPU传输）
    use_amp           = True    # 是否启用混合精度训练
    amp_dtype         = "bf16"           # "bf16" 或 "fp16"
    use_tf32          = True    # 是否启用TF32精度
    use_torch_compile = False   # 是否启用TorchCompile加速
    print_step_interval = 10    # 每多少步打印一次日志

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