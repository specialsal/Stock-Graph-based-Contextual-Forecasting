# coding: utf-8
"""
滚动回测专用配置（按窗口 step_weeks 片段拼接）
"""
from dataclasses import dataclass
from pathlib import Path
import torch
from datetime import datetime

@dataclass
class BTRollingConfig:
    # 与训练一致的 run_name（用于定位模型目录）
    run_name      = "TGF-context"
    # run_name      = "TGF-abl_noGNN"
    # run_name      = "TGF-abl_noFiLM"
    # run_name      = "TGF-abl_noTR"
    # run_name      = "TGF-abl_noGNN_noFiLM"

    # 路径
    model_dir     = Path(f"./models/model_{run_name}")
    backtest_dir  = Path(f"./backtest_rolling/{run_name}")

    # 回测区间（周五采样日）
    bt_start_date = "2017-02-10"
    bt_end_date   = datetime.today().strftime("%Y-%m-%d")
    # bt_end_date   = "2025-10-31"
    # bt_start_date = "2015-01-01"
    # bt_end_date   = "2015-12-31"

    # 运行名（用于输出文件命名）
    run_name_out  = run_name

    # 数据路径
    data_dir       = Path("./data")
    processed_dir  = data_dir / "processed"
    raw_dir        = data_dir / "raw"
    feat_file      = processed_dir / "features_daily.h5"
    ctx_file       = processed_dir / "context_features.parquet"
    label_file     = processed_dir / "weekly_labels.parquet"
    industry_map_file = raw_dir / "stock_industry_map.csv"
    trading_day_file  = raw_dir / "trading_day.csv"
    price_day_file     = raw_dir / "stock_price_day.parquet"
    stock_info_file    = raw_dir / "stock_info.csv"
    is_suspended_file  = raw_dir / "is_suspended.csv"
    is_st_file         = raw_dir / "is_st_stock.csv"

    # 回测模式
    mode           = "long"
    long_weight    = 1.0
    short_weight   = 1.0

    # 分组与持仓控制
    top_pct        = 0.05
    bottom_pct     = 0.05
    min_n_stocks   = 8 # 50
    max_n_stocks   = 8 #300

    # 成本与滑点（非对称）
    # 注意：bps 为基点（万分之一）。例如 3 表示万三（0.03%）
    buy_fee_bps    = 5   # 买入手续费：万零点95
    sell_fee_bps   = 10   # 卖出手续费：万5点95
    slippage_bps   = 5   # 双边对称滑点：万五（买卖两侧都加）

    # 过滤与样本要求（与训练一致，可按需启用）
    enable_filters     = True
    ipo_cut_days       = 120
    suspended_exclude  = True
    st_exclude         = True
    min_daily_turnover = 5e6
    allow_missing_info = False

    # 板块开关（与训练一致）
    include_star_market = False
    include_chinext = False
    include_bse = False
    include_neeq = False

    # 周内止盈/止损参数
    enable_intraweek_stops: bool = False
    tp_price_ratio: float = 99
    sl_price_ratio: float = 99

    # 组合权重与过滤（从代码中上移到配置）
    weight_mode: str = "equal"  # "equal" 或 "score"
    filter_negative_scores_long: bool = False

    # 行业中性化开关与参数（新增）
    neutralize_enable: bool = False               # 是否开启行业中性化
    neutralize_method: str = "ols_resid"          # "ols_resid" 或 "group_demean"（预留）
    neutralize_add_intercept: bool = True         # OLS 设计矩阵是否包含截距
    neutralize_min_group_size: int = 1            # 行业内最小样本数（用于健壮性判定）
    neutralize_clip_pct: float = 0.0              # 对原始分数 winsorize 百分位（0~0.5），0 表示不裁剪

    # 调仓频率（新增）：1=每周；2=双周；3=三周
    rebalance_every_k: int = 1
    
    # 设备
    device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BT_ROLL_CFG = BTRollingConfig()