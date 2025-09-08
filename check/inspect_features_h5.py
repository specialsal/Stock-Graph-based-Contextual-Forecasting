# coding: utf-8
import os
import sys
import h5py
import numpy as np
import pandas as pd
from pathlib import Path

try:
    # 若你在同一工程内，可直接使用 CFG 来默认定位 h5 文件
    from config import CFG
    DEFAULT_H5 = Path(CFG.feat_file)
except Exception:
    DEFAULT_H5 = None

def summarize_array(name, arr, max_print=5):
    print(f"  [{name}] dtype={arr.dtype}, shape={arr.shape}")
    # 仅对浮点数组做 NaN/Inf 检查
    if np.issubdtype(arr.dtype, np.floating):
        n_nan = np.isnan(arr).sum()
        n_inf = np.isinf(arr).sum()
        print(f"    NaN count={int(n_nan)}, Inf count={int(n_inf)}")
        if arr.size > 0:
            try:
                sample_vals = arr.ravel()[::max(1, arr.size // max_print)][:max_print]
                print(f"    sample values: {sample_vals}")
            except Exception:
                pass

def inspect_h5(h5_path: Path, fast=False):
    if not h5_path.exists():
        print(f"[错误] 文件不存在: {h5_path}")
        return

    print(f"打开 H5: {h5_path}")
    with h5py.File(h5_path, "r") as f:
        # 顶层属性
        print("\n[顶层属性 attrs]")
        for k, v in f.attrs.items():
            try:
                if isinstance(v, bytes):
                    v = v.decode("utf-8", errors="ignore")
                elif isinstance(v, np.ndarray) and v.dtype.kind in ('S','U','O'):
                    # 打印前几项
                    to_show = v[:5]
                    print(f"  {k}: ndarray(len={len(v)}), sample={to_show}")
                    continue
            except Exception:
                pass
            print(f"  {k}: {v}")

        # 列出所有组
        print("\n[组列表]")
        groups = sorted(list(f.keys()))
        print(f"  组数量={len(groups)}")
        if len(groups) == 0:
            print("  (空文件，无任何组)")
            return

        # 收集每组日期与基本信息
        dates = []
        date_to_group = {}
        for gname in groups:
            g = f[gname]
            d_attr = g.attrs.get("date", None)
            try:
                d_str = d_attr.decode("utf-8") if isinstance(d_attr, (bytes, bytearray)) else d_attr
                d_ts = pd.Timestamp(d_str) if d_str is not None else None
            except Exception:
                d_ts = None
            dates.append(d_ts)
            date_to_group[d_ts] = gname

        # 打印日期覆盖、重复情况
        dates_clean = [d for d in dates if d is not None]
        print("\n[日期覆盖]")
        if dates_clean:
            print(f"  最早日期={min(dates_clean).date()}, 最晚日期={max(dates_clean).date()}, 共有组={len(dates_clean)}")
            # 检查重复日期
            vc = pd.Series(dates_clean).value_counts()
            dup_dates = vc[vc > 1]
            if len(dup_dates) > 0:
                print("  [警告] 存在重复日期组：")
                for d, cnt in dup_dates.items():
                    print(f"    {d.date()} x {cnt}")
            else:
                print("  无重复日期组")
        else:
            print("  未发现有效日期属性（attrs['date']），请检查写入逻辑。")

        # 随机/首末抽样若干组进行详细检查
        to_check = []
        if dates_clean:
            dates_sorted = sorted(dates_clean)
            if fast:
                # 快速模式：只检查首尾各1个
                samples = [dates_sorted[0], dates_sorted[-1]]
            else:
                # 默认检查首2、尾2、以及中位1个（去重）
                picks = set()
                if len(dates_sorted) >= 1:
                    picks.add(dates_sorted[0])
                    picks.add(dates_sorted[-1])
                if len(dates_sorted) >= 4:
                    picks.add(dates_sorted[1])
                    picks.add(dates_sorted[-2])
                picks.add(dates_sorted[len(dates_sorted)//2])
                samples = sorted(picks)
            to_check = samples

        print("\n[详细检查若干组]")
        for d in to_check:
            gname = date_to_group.get(d, None)
            if gname is None:
                continue
            g = f[gname]
            print(f"\n  Group: {gname}, date={d.date() if d is not None else None}")
            # stocks
            if "stocks" in g:
                stocks = g["stocks"][...]
                try:
                    # 字符数组转 Python 列表
                    if stocks.dtype.kind == 'S':
                        stocks = [s.decode("utf-8", errors="ignore") for s in stocks]
                    else:
                        stocks = stocks.astype(str).tolist()
                except Exception:
                    stocks = list(stocks)
                print(f"    stocks: count={len(stocks)}")
                if len(stocks) > 0:
                    print(f"    sample stocks: {stocks[:5]}")
            else:
                print("    [缺失] dataset 'stocks' 不存在")

            # factor
            if "factor" in g:
                ds = g["factor"]
                print(f"    factor dataset: shape={ds.shape}, dtype={ds.dtype}")
                # 统计N,T,C
                if len(ds.shape) == 3:
                    N, T, C = ds.shape
                    print(f"    -> N(股票数)={N}, T(时间窗口)={T}, C(因子数)={C}")
                # 抽样检查数值
                try:
                    # 只取小片段避免内存负担
                    n_sample = min(3, ds.shape[0])
                    t_sample = min(3, ds.shape[1]) if ds.shape[1] >= 1 else 0
                    c_sample = min(5, ds.shape[2]) if len(ds.shape) == 3 else 0
                    if n_sample > 0:
                        sample_block = ds[0:n_sample, 0:t_sample, 0:c_sample]
                        summarize_array("factor(sample)", sample_block)
                    # 全量 NaN/Inf 统计（逐块以节约内存）
                    n_nan = 0
                    n_inf = 0
                    step = max(1, ds.shape[0] // 8)
                    for i in range(0, ds.shape[0], step):
                        chunk = ds[i:i+step]
                        if np.issubdtype(chunk.dtype, np.floating):
                            n_nan += int(np.isnan(chunk).sum())
                            n_inf += int(np.isinf(chunk).sum())
                    print(f"    factor 全量 NaN={n_nan}, Inf={n_inf}")
                except Exception as e:
                    print(f"    [读取factor异常] {e}")
            else:
                print("    [缺失] dataset 'factor' 不存在")

        # 额外：打印 factor_cols（如有）
        if 'factor_cols' in f.attrs:
            fac_cols = f.attrs['factor_cols']
            try:
                if isinstance(fac_cols, np.ndarray) and fac_cols.dtype.kind == 'S':
                    fac_cols = [x.decode('utf-8', errors='ignore') for x in fac_cols]
                elif isinstance(fac_cols, (bytes, bytearray)):
                    fac_cols = [fac_cols.decode('utf-8', errors='ignore')]
            except Exception:
                pass
            print("\n[factor_cols 概览]")
            if isinstance(fac_cols, (list, tuple, np.ndarray)):
                print(f"  因子数={len(fac_cols)}")
                print(f"  sample: {list(fac_cols)[:10]}")
            else:
                print(f"  factor_cols: {fac_cols}")

    print("\n[检查完成]")

if __name__ == "__main__":
    # 用法：
    #   python inspect_features_h5.py /path/to/features_daily.h5
    #   python inspect_features_h5.py  (使用 CFG.feat_file 作为默认)
    if len(sys.argv) > 1:
        h5_file = Path(sys.argv[1])
    else:
        if DEFAULT_H5 is None:
            print("请在命令行传入 H5 路径，或确保 config.CFG 可用。")
            sys.exit(1)
        h5_file = Path(DEFAULT_H5)
    # fast=True 仅检查首尾；fast=False 更全面抽样
    inspect_h5(h5_file, fast=False)