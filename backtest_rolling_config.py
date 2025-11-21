# coding: utf-8
"""
滚动回测专用配置（按窗口 step_weeks 片段拼接）

本版本在原有基础上新增：
- 市值中性化相关配置（mkt_neutral_*），用于在 optimize_position 中对分数做“市值分箱去均值”：
  - mkt_neutral_enable: 是否开启市值中性化
  - mkt_neutral_n_bins: log(市值) 的分箱数量（如 5 档或 10 档）
  - mkt_neutral_clip_pct: 对 log(市值) 的 winsorize 百分位（0~0.5），0 表示不裁剪

注意：
- 行业中性化与市值中性化是两个可独立开关：
  1) neutralize_enable=True,  mkt_neutral_enable=False  -> 仅行业中性化
  2) neutralize_enable=False, mkt_neutral_enable=True   -> 仅市值中性化
  3) neutralize_enable=True,  mkt_neutral_enable=True   -> 先行业中性化，再在残差上做市值中性化
  4) 两者都 False -> 不做中性化，直接用原始 score 排序/赋权
"""

from dataclasses import dataclass
from pathlib import Path
import torch
from datetime import datetime

@dataclass
class BTRollingConfig:
    # 与训练一致的 run_name（用于定位模型目录）
    run_name      = "TGF-model"
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
    price_day_file        = raw_dir / "stock_price_day.parquet"
    stock_info_file       = raw_dir / "stock_info.csv"
    is_suspended_file     = raw_dir / "is_suspended.csv"
    is_st_file            = raw_dir / "is_st_stock.csv"
    # 新增：市值所在文件（与 Config 中的 price_fundamental_file 对应）
    price_fundamental_file = raw_dir / "stock_fundamental_day.parquet"

    # 回测模式
    mode           = "long"
    long_weight    = 1.0
    short_weight   = 1.0

    # 分组与持仓控制
    top_pct        = 0.05
    bottom_pct     = 0.05
    min_n_stocks   = 8   # 最少持仓数
    max_n_stocks   = 8   # 最大持仓数

    # 成本与滑点（非对称）
    # 注意：bps 为基点（万分之一）。例如 3 表示万三（0.03%）
    buy_fee_bps    = 5    # 买入手续费：万零点5
    sell_fee_bps   = 10   # 卖出手续费：万1
    slippage_bps   = 5    # 双边对称滑点：万5（买卖两侧都加）

    # 过滤与样本要求（与训练一致，可按需启用）
    enable_filters     = True
    ipo_cut_days       = 120
    suspended_exclude  = True
    st_exclude         = True
    min_daily_turnover = 5e6
    allow_missing_info = False

    # 板块开关（与训练一致）
    include_star_market = False
    include_chinext     = False
    include_bse         = False
    include_neeq        = False

    # 周内止盈/止损参数
    enable_intraweek_stops: bool = True
    tp_price_ratio: float = 0.15
    sl_price_ratio: float = 0.01
    cooldown_days: float = 1

    # 组合权重与过滤（从代码中上移到配置）
    weight_mode: str = "equal"  # "equal" 或 "score"
    filter_negative_scores_long: bool = False

    # 行业中性化开关与参数（已有）
    neutralize_enable: bool = False               # 是否开启行业中性化
    neutralize_method: str = "ols_resid"          # "ols_resid" 或 "group_demean"（当前实现主要用 ols_resid）
    neutralize_add_intercept: bool = True         # OLS 设计矩阵是否包含截距
    neutralize_min_group_size: int = 1            # 行业内最小样本数（预留，当前未强制使用）
    neutralize_clip_pct: float = 0.0              # 对原始分数 winsorize 百分位（0~0.5），0 表示不裁剪

    # 调仓频率（新增）：1=每周；2=双周；3=三周
    rebalance_every_k: int = 1

    # ========= 市值中性化相关配置（新增） =========
    # 是否对分数做“市值中性化”（按 log(市值) 分箱，箱内去均值）
    mkt_neutral_enable: bool = False
    # log(市值) 的分箱数量：例如 5 表示按分位数切成 5 档
    mkt_neutral_n_bins: int = 3
    # 对 log(市值) 做 winsorize 的百分位，0 表示不裁剪；例如 0.01 表示 1% / 99% 裁剪
    mkt_neutral_clip_pct: float = 0

    # 设备
    device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")


BT_ROLL_CFG = BTRollingConfig()