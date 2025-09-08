# coding: utf-8
"""
数据一致性校验脚本
- 校验 features_daily.h5 / weekly_labels.parquet / context_features.parquet 的结构与对齐
- 输出每个周五的覆盖情况与潜在问题
运行方式：python check_data_integrity.py
"""
import os
import math
import json
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import h5py

# 项目内模块
from config import CFG
from utils import load_calendar, weekly_fridays, load_industry_map

def read_h5_meta_all(h5_path: Path):
    if not Path(h5_path).exists():
        raise FileNotFoundError(f"H5 不存在：{h5_path}")
    dates = []
    groups = []
    with h5py.File(h5_path, "r") as h5:
        if 'factor_cols' in h5.attrs:
            factor_cols = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in np.array(h5.attrs['factor_cols'])]
        else:
            factor_cols = None
        for k in h5.keys():
            if not k.startswith("date_"):
                continue
            g = h5[k]
            d_str = g.attrs.get('date', None)
            if d_str is None:
                continue
            if isinstance(d_str, bytes):
                d_str = d_str.decode('utf-8')
            dt = pd.to_datetime(d_str)
            dates.append(dt)
            groups.append(k)
    df = pd.DataFrame({"group": groups, "date": dates}).sort_values("date").reset_index(drop=True)
    return df, factor_cols

def scan_h5_groups(h5_path: Path, sample_limit: int = 5):
    """
    返回：
      - meta_df: 各组日期与名称
      - head_samples: 前若干组的统计信息
      - per_date_counts: 每组股票数
      - shape_info: 统一的 (T, C) 维度与不一致告警
    """
    meta_df, factor_cols = read_h5_meta_all(h5_path)
    shape_T = None
    shape_C = None
    head_samples = []
    per_date_counts = {}
    issues = []

    with h5py.File(h5_path, "r") as h5:
        for i, row in meta_df.iterrows():
            gk = row["group"]
            dt = row["date"]
            if gk not in h5:
                issues.append(f"缺失组 {gk}")
                continue
            g = h5[gk]
            if "stocks" not in g or "factor" not in g:
                issues.append(f"{gk} 缺少 stocks 或 factor 数据集")
                continue
            stocks = g["stocks"][:].astype(str)
            X = np.asarray(g["factor"][:])  # [N,T,C]
            if X.ndim != 3:
                issues.append(f"{gk} factor 维度异常，期望 [N,T,C]，得到 {X.shape}")
                continue
            N, T, C = X.shape
            per_date_counts[pd.Timestamp(dt)] = int(N)
            if shape_T is None: shape_T = T
            if shape_C is None: shape_C = C
            if (T != shape_T) or (C != shape_C):
                issues.append(f"{gk} 的 (T,C)=({T},{C}) 与前者不一致，期望 ({shape_T},{shape_C})")
            if len(head_samples) < sample_limit:
                head_samples.append({
                    "group": gk, "date": str(dt.date()), "N": int(N), "T": int(T), "C": int(C),
                    "first_stocks": stocks[: min(5, len(stocks))].tolist()
                })

    shape_info = {"T": shape_T, "C": shape_C, "factor_cols": factor_cols}
    # 重复日期检测
    dup_dates = meta_df["date"][meta_df["date"].duplicated()].tolist()
    if dup_dates:
        issues.append(f"H5 组中存在重复日期（应避免）：{sorted(set([str(d.date()) for d in dup_dates]))}")

    return meta_df, head_samples, per_date_counts, shape_info, issues

def load_labels(label_path: Path):
    if not Path(label_path).exists():
        raise FileNotFoundError(f"标签文件不存在：{label_path}")
    df = pd.read_parquet(label_path)
    if not isinstance(df.index, pd.MultiIndex) or set(df.index.names) != set(["date","stock"]):
        raise ValueError(f"标签索引需为 MultiIndex(date, stock)，当前={df.index.names}")
    if "next_week_return" not in df.columns:
        raise ValueError("标签需包含列 next_week_return")
    # 统计每个周五的样本数
    counts = df.groupby(level="date").size().to_dict()
    dates = sorted(df.index.get_level_values("date").unique())
    return df, counts, dates

def load_context(ctx_path: Path):
    if not Path(ctx_path).exists():
        raise FileNotFoundError(f"上下文文件不存在：{ctx_path}")
    df = pd.read_parquet(ctx_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("context_features 索引需为 DatetimeIndex（周五）")
    nan_ratio = float(df.isna().sum().sum()) / max(1, (df.shape[0] * df.shape[1]))
    return df, nan_ratio

def check_industry_map(ind_csv: Path, sample_stocks: list):
    if not Path(ind_csv).exists():
        return {"exists": False, "covered_ratio": None, "unknown": [], "n_total": len(sample_stocks)}
    ind_map = load_industry_map(ind_csv)
    unknown = [s for s in sample_stocks if s not in ind_map]
    covered = len(sample_stocks) - len(unknown)
    ratio = covered / max(1, len(sample_stocks))
    return {"exists": True, "covered_ratio": ratio, "unknown": unknown[:50], "n_total": len(sample_stocks)}

def intersect_dates(h5_dates, label_dates, ctx_dates):
    set_h5 = set(h5_dates)
    set_lb = set(label_dates)
    set_ctx = set(ctx_dates)
    inter = sorted(set_h5 & set_lb & set_ctx)
    only_h5  = sorted(set_h5 - set_lb - set_ctx)
    h5_not_label = sorted(set_h5 - set_lb)
    h5_not_ctx   = sorted(set_h5 - set_ctx)
    return inter, only_h5, h5_not_label, h5_not_ctx

def compute_effective_samples_per_date(h5_path: Path, valid_dates: list, label_df: pd.DataFrame, ctx_df: pd.DataFrame):
    """
    近似估计每个周五最终可进入训练的样本数量：
    - 仅依据 H5 的 stocks ∩ 标签存在性；不考虑过滤器（停牌/成交额等），也不考虑行业映射缺失
    """
    perdate_effective = {}
    with h5py.File(h5_path, "r") as h5:
        # 建立 {date -> group_names} 映射
        date2groups = defaultdict(list)
        for k in h5.keys():
            if not k.startswith("date_"):
                continue
            d_str = h5[k].attrs.get('date', None)
            if d_str is None:
                continue
            if isinstance(d_str, bytes):
                d_str = d_str.decode('utf-8')
            dt = pd.to_datetime(d_str)
            date2groups[dt].append(k)

        for d in valid_dates:
            d = pd.Timestamp(d)
            if d not in date2groups:
                perdate_effective[d] = 0
                continue
            # 正常情况一个日期只有一个组，若有多个组（不建议），取并集
            total = 0
            for gk in date2groups[d]:
                stocks = h5[gk]["stocks"][:].astype(str)
                # 与标签对齐
                idx_tuples = pd.MultiIndex.from_product([[d], stocks], names=["date","stock"])
                mask = idx_tuples.isin(label_df.index)
                total += int(mask.sum())
            perdate_effective[d] = total
    return perdate_effective

def main():
    print("=== 1) 读取交易周五全集 ===")
    cal = load_calendar(CFG.trading_day_file)
    fridays = weekly_fridays(cal)
    fridays = fridays[(fridays >= pd.Timestamp(CFG.start_date)) & (fridays <= pd.Timestamp(CFG.end_date))]
    print(f"交易日历周五数：{len(fridays)}  范围：{fridays.min().date()} ~ {fridays.max().date()}")

    print("\n=== 2) 扫描 H5 特征仓 ===")
    h5_path = Path(CFG.feat_file)
    meta_df, head_samples, h5_counts, shape_info, h5_issues = scan_h5_groups(h5_path)
    print(f"H5 组数：{len(meta_df)}，示例前{len(head_samples)}组：")
    for r in head_samples:
        print("  -", r)
    if shape_info["factor_cols"] is not None:
        print(f"因子维度 C={shape_info['C']}，窗口 T={shape_info['T']}，因子列数={len(shape_info['factor_cols'])}")
    if h5_issues:
        print("H5 发现问题：")
        for msg in h5_issues:
            print("  [H5] ", msg)

    print("\n=== 3) 读取标签 ===")
    label_df, label_counts, label_dates = load_labels(Path(CFG.label_file))
    print(f"标签周五数：{len(label_dates)}  标签总行数：{len(label_df)}")
    # 打印部分统计
    if len(label_counts) > 0:
        s = pd.Series(label_counts).sort_index()
        print(f"  标签每日样本数（前5行）:\n{s.head(5)}")

    print("\n=== 4) 读取上下文 ===")
    ctx_df, ctx_nan_ratio = load_context(Path(CFG.processed_dir / "context_features.parquet"))
    print(f"上下文周五数：{len(ctx_df)}  列数：{ctx_df.shape[1]}  NaN占比：{ctx_nan_ratio:.2%}")

    print("\n=== 5) 日期集合对齐 ===")
    inter_dates, only_h5, h5_not_label, h5_not_ctx = intersect_dates(
        meta_df["date"].tolist(), label_dates, ctx_df.index.tolist()
    )
    print(f"三者交集（可用于训练）的周五数：{len(inter_dates)}")
    if len(inter_dates) > 0:
        print(f"  范围：{pd.Timestamp(min(inter_dates)).date()} ~ {pd.Timestamp(max(inter_dates)).date()}")
    if h5_not_label:
        print(f"H5 有而标签无的周五（前10）：{[str(pd.Timestamp(d).date()) for d in h5_not_label[:10]]} ... 共{len(h5_not_label)}")
    if h5_not_ctx:
        print(f"H5 有而上下文无的周五（前10）：{[str(pd.Timestamp(d).date()) for d in h5_not_ctx[:10]]} ... 共{len(h5_not_ctx)}")

    print("\n=== 6) 估计每个交集周五的有效样本数（与标签对齐后） ===")
    eff_counts = compute_effective_samples_per_date(h5_path, inter_dates, label_df, ctx_df)
    if len(eff_counts) > 0:
        se = pd.Series(eff_counts).sort_index()
        print(f"  交集每日可用样本数（前5行）:\n{se.head(5)}")
        print(f"  平均/中位数样本数：{se.mean():.1f} / {se.median():.1f}")

    print("\n=== 7) 行业映射覆盖率 ===")
    # 从 H5 取一部分股票样本检测行业覆盖率
    sample_stocks = set()
    with h5py.File(h5_path, "r") as h5:
        for k in list(h5.keys())[:10]:
            if "stocks" in h5[k]:
                sample_stocks.update(h5[k]["stocks"][:].astype(str).tolist())
    ind_info = check_industry_map(Path(CFG.industry_map_file), sorted(sample_stocks))
    if not ind_info["exists"]:
        print(f"行业映射缺失：{CFG.industry_map_file}")
    else:
        print(f"行业映射覆盖率（基于前10组采样的股票）：{ind_info['covered_ratio']*100:.2f}% / 样本数={ind_info['n_total']}")
        if ind_info["unknown"]:
            print(f"  未覆盖样本（前50）：{ind_info['unknown']}")

    print("\n=== 8) 潜在问题与建议 ===")
    tips = []
    if shape_info["T"] is None or shape_info["C"] is None:
        tips.append("H5 中没有有效的 factor 组，确认 feature_engineering 是否已生成。")
    if h5_issues:
        tips.append("修复 H5 重复日期/维度不一致问题，确保每个周五只有一个组且 (T,C) 一致。")
    if len(inter_dates) == 0:
        tips.append("三者交集为空：检查 context 与 label 是否按交易周五覆盖，并与 H5 对齐。")
    if ctx_nan_ratio > 0:
        tips.append(f"上下文含 NaN（{ctx_nan_ratio:.2%}），训练前可考虑填充或剔除相应日期。")
    missing_in_fridays = sorted(set(fridays) - set(meta_df["date"]))
    if missing_in_fridays:
        tips.append(f"H5 未覆盖全部日历周五（前10缺失）：{[str(d.date()) for d in missing_in_fridays[:10]]}")
    # 打印建议
    if tips:
        for t in tips:
            print("  -", t)
    else:
        print("未发现明显问题，可以开始训练。")

    print("\n校验完成。")

if __name__ == "__main__":
    main()