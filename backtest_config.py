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
    model_dir      = processed_dir / "models"
    industry_map_file = raw_dir / "stock_industry_map.csv"
    trading_day_file  = raw_dir / "trading_day.csv"

    # 选择回测使用的模型：
    # 可填：具体文件名（例如 "model_20200103.pth" / "model_best_20200103.pth"
    #      或固定别名 "best_overall.pth" / "best_recent_5.pth"）
    model_name     = "best_overall.pth"

    # 回测区间（周五采样日）
    bt_start_date  = "2011-01-01"
    bt_end_date    = datetime.today().strftime("%Y-%m-%d")

    # 运行名（用于输出文件命名）
    run_name       = "demo"

    # 回测模式
    # mode = "long"  仅做多
    # mode = "ls"    多空对冲（多头等权，空头等权，净敞口可通过 long_weight/short_weight 控制）
    mode           = "ls"
    long_weight    = 1.0
    short_weight   = 1.0

    # 分组与持仓控制
    top_pct        = 0.1    # 做多比例（0~1），仅在当周股票数足够时生效
    bottom_pct     = 0.1    # 做空比例（0~1），仅在 mode=="ls" 时使用
    min_n_stocks   = 20     # 每周最少持仓数量（若不足则本周空仓）
    max_n_stocks   = 200    # 每边最多持仓数量（多/空各自限制）

    # 交易相关假设
    # rebalance_day: 使用每个周五的打分，在下一个交易周的第一个可交易日开盘成交（简化为当周收盘->下周收盘收益）
    # 我们用标签同款“周五对齐法”：从周五到下一个周五的收盘涨跌作为一周收益
    # 真实滑点与手续费请按需在这里增加假设参数
    slippage_bps   = 0.0    # 单边滑点，基点
    fee_bps        = 0.0    # 单边手续费，基点

    # 设备
    device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BT_CFG = BTConfig()