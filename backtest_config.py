# coding: utf-8
"""
回测专用配置
"""
from dataclasses import dataclass
from pathlib import Path
import torch
from datetime import datetime

@dataclass
class BTConfig:
    # 基础路径（沿用训练配置目录结构）
    data_dir       = Path("./data")
    processed_dir  = data_dir / "processed"
    raw_dir        = data_dir / "raw"
    feat_file      = processed_dir / "features_daily.h5"
    ctx_file       = processed_dir / "context_features.parquet"
    label_file     = processed_dir / "weekly_labels.parquet"  # 可选
    model_dir      = processed_dir / "models_gat_1_1"
    industry_map_file = raw_dir / "stock_industry_map.csv"
    trading_day_file  = raw_dir / "trading_day.csv"

    backtest_dir = Path("./backtest")

    # 为回测筛选加载原始数据（与训练一致）
    price_day_file     = raw_dir / "stock_price_day.parquet"
    stock_info_file    = raw_dir / "stock_info.csv"
    is_suspended_file  = raw_dir / "is_suspended.csv"
    is_st_file         = raw_dir / "is_st_stock.csv"

    # 选择回测使用的模型：
    # 可填：具体文件名（例如 "model_20200103.pth" / "model_best_20200103.pth"
    #      或固定别名 "best_overall.pth" / "best_recent_5.pth"）
    model_name     = "model_best_20240403.pth"

    # 回测区间（周五采样日）
    bt_start_date  = "2024-04-03"
    bt_end_date    = datetime.today().strftime("%Y-%m-%d")

    # 运行名（用于输出文件命名）
    run_name       = "demo"

    # 回测模式
    # mode = "long"  仅做多
    # mode = "ls"    多空对冲（多头等权，空头等权，净敞口可通过 long_weight/short_weight 控制）
    mode           = "long"
    long_weight    = 1.0
    short_weight   = 1.0

    # 分组与持仓控制
    top_pct        = 0.1    # 做多比例（0~1），仅在当周股票数足够时生效
    bottom_pct     = 0.1   # 做空比例（0~1），仅在 mode=="ls" 时使用
    min_n_stocks   = 20     # 每周最少持仓数量（若不足则本周空仓）
    max_n_stocks   = 300    # 每边最多持仓数量（多/空各自限制）

    # 交易相关假设
    slippage_bps   = 0.005    # 单边滑点，基点
    fee_bps        = 0.001    # 单边手续费，基点

    # 样本过滤（对回测当周可选股票集合进行过滤，口径与训练一致）
    enable_filters     = True     # 是否启用筛选
    ipo_cut_days       = 120      # IPO满多少天才纳入（自然日）
    suspended_exclude  = True     # 排除停牌
    st_exclude         = True     # 排除 ST
    min_daily_turnover = 5e6      # 当日最低成交额阈值（单位与原始数据一致）
    allow_missing_info = False    # 缺少基础信息是否保留

    # 板块开关（为 False 则剔除该板块股票）
    include_star_market = False  # 科创板（688/689.XSHG）
    include_chinext = False  # 创业板（300/301.XSHE）
    include_bse = False  # 北交所（*.XBEI / *.XBSE）
    include_neeq = False  # 新三板（*.XNE / *.XNEE / *.XNEQ / *.XNEX）

    # 设备
    device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BT_CFG = BTConfig()