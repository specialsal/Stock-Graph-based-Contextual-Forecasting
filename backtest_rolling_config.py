# coding: utf-8
"""
滚动回测专用配置（按窗口 step_weeks 片段拼接）
- 模型目录按 run_name 组织：./models/model_{run_name}
- 回测输出目录：./backtest_rolling/{run_name}
- 其它参数尽量与 backtest_config 保持一致
"""
from dataclasses import dataclass
from pathlib import Path
import torch
from datetime import datetime

@dataclass
class BTRollingConfig:
    # 与训练一致的 run_name（用于定位模型目录）
    run_name      = "tr1gat1win50"

    # 路径
    model_dir     = Path(f"./models/model_{run_name}")
    backtest_dir  = Path(f"./backtest_rolling/{run_name}")

    # 回测区间（周五采样日）
    bt_start_date = "2017-02-10"
    bt_end_date   = datetime.today().strftime("%Y-%m-%d")

    # 运行名（用于输出文件命名）
    run_name_out  = run_name

    # 基础路径（沿用训练配置目录结构）
    data_dir       = Path("./data")
    processed_dir  = data_dir / "processed"
    raw_dir        = data_dir / "raw"
    feat_file      = processed_dir / "features_daily.h5"
    ctx_file       = processed_dir / "context_features.parquet"
    label_file     = processed_dir / "weekly_labels.parquet"  # 可选，用于计算收益
    industry_map_file = raw_dir / "stock_industry_map.csv"
    trading_day_file  = raw_dir / "trading_day.csv"
    # 为回测筛选加载原始数据（与训练一致）
    price_day_file     = raw_dir / "stock_price_day.parquet"
    stock_info_file    = raw_dir / "stock_info.csv"
    is_suspended_file  = raw_dir / "is_suspended.csv"
    is_st_file         = raw_dir / "is_st_stock.csv"

    # 回测模式
    # mode = "long"  仅做多
    # mode = "ls"    多空对冲（多头等权，空头等权，净敞口可通过 long_weight/short_weight 控制）
    mode           = "long"
    long_weight    = 1.0
    short_weight   = 1.0

    # 分组与持仓控制
    top_pct        = 0.05    # 做多比例（0~1）
    bottom_pct     = 0.05    # 做空比例（0~1），仅在 mode=="ls" 时使用
    min_n_stocks   = 50     # 每周最少持仓数量（若不足则本周空仓) 50
    max_n_stocks   = 300    # 每边最多持仓数量（多/空各自限制）300

    # 交易相关假设
    slippage_bps   = 5  # 单边滑点，基点
    fee_bps        = 3  # 单边手续费，基点

    # 样本过滤（与训练一致）
    enable_filters     = True
    ipo_cut_days       = 120
    suspended_exclude  = True
    st_exclude         = True
    min_daily_turnover = 5e6
    allow_missing_info = False

    # 板块开关（与训练一致）
    include_star_market = False  # 科创板
    include_chinext = False      # 创业板
    include_bse = False          # 北交所
    include_neeq = False         # 新三板

    # 设备
    device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BT_ROLL_CFG = BTRollingConfig()