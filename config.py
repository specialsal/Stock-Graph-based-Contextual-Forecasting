# ====================== config.py ======================
# coding: utf-8
"""
全局配置 —— 提速版（含图模块简化、AMP/Loader优化、指标与模型选择）
说明：
- 每个训练窗口在线拟合当期训练集 Scaler（避免未来信息泄漏）
- ranking_weight: 训练损失 = w*(1 - Pearson) + (1-w)*PairwiseRanking
- test_weeks: 每个窗口引入测试集（单位=周，默认52周）
- 保存策略：每窗口保存 best（按测试集风险调整分数）与 last，并登记到 registry_file
- 本版已移除“全局最优(best_overall)/最近N最优(best_recent_N)”的输出逻辑相关配置
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
    model_dir     = Path("./models")                      # 模型输出目录
    feat_file     = processed_dir / "features_daily.h5"
    label_file    = processed_dir / "weekly_labels.parquet"
    universe_file = processed_dir / "universe.pkl"
    registry_file = model_dir / "model_registry.csv"      # 窗口登记表（保留）

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
    ram_accel_enable     = True     # 内存大的机器可开启
    ram_accel_mem_cap_gb = 48       # RAM 加速单窗口的内存上限（GB）；超过将自动回退为逐组加载

    # -------- 股票筛选（可选） --------
    enable_filters      = True       # 是否启用筛选（关闭则与旧逻辑一致）
    ipo_cut_days        = 120        # 新股上市满多少天才纳入（自然日）
    suspended_exclude   = True       # 排除停牌
    st_exclude          = True       # 排除 ST
    min_daily_turnover  = 5e6        # 当日最低成交额（单位与原始数据一致，如人民币元）
    allow_missing_info  = False      # 缺少基础信息的股票是否保留（默认丢弃）

    # -------- 特征窗口 --------
    max_lookback = 120  # 最大回溯天数（确保足够覆盖所有日频特征）在增量更新时使用
    daily_window = 20   # 日频特征的时间窗口大小

    # -------- 滚动窗参数（单位=周）--------
    train_years = 5     # 训练年数（以年计）；实际以 5*52 周计算
    val_weeks   = 52    # 验证周数
    test_weeks  = 52    # 测试周数
    step_weeks  = 52    # 步长（周）
    start_date  = "2011-01-01"
    end_date    = datetime.today().strftime("%Y-%m-%d")  # 默认到今天为止

    # -------- 模型超参 --------
    hidden        = 64  # 隐藏层维度
    ind_emb       = 16  # 行业嵌入维度
    ctx_dim       = 21  # 上下文特征维度
    tr_layers     = 1   # Transformer 层数
    gat_layers    = 1   # GAT 层数
    graph_type    = "gat"        # 图模块类型 "mean" 或 "gat"
    lr            = 3e-4
    weight_decay  = 1e-2
    epochs_warm   = 10  # 每个窗口训练轮数

    # -------- 训练细节 --------
    batch_size        = 8196
    grad_accum_steps  = 1
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

    # -------- 选择与记录（风险调整权重） --------
    score_alpha     = 0.5       # 风险调整参数：score = mean - alpha * std（用于每窗口best的评判）
    ranking_weight  = 0.5       # loss = w*(1 - Pearson) + (1-w)*pairwise_rank

    # -------- 板块开关 --------
    include_star_market = False  # 科创板（688/689.XSHG）
    include_chinext = False      # 创业板（300/301.XSHE）
    include_bse = False          # 北交所（*.XBEI / *.XBSE）
    include_neeq = False         # 新三板（*.XNE / *.XNEE / *.XNEQ / *.XNEX）

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