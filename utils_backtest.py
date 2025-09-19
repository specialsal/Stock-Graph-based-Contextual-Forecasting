# coding: utf-8
"""
utils_backtest.py
公共工具：指标计算、回撤、年度切分、窗口跨度恢复、绘图、超额 NAV 生成等
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ========== 基础 I/O ==========

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def read_nav_csv(path: Path) -> pd.DataFrame:
    """
    读取 nav_{run_name}.csv，期望包含列：
      - ret_total, nav, n_long（必有）
      - 可选：ret_long, ret_short, n_short
    index: 日期（如果不是索引，则尝试将第一列解析为日期索引）
    """
    df = pd.read_csv(path)
    # 尝试识别日期列
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    else:
        # 若第一列是日期
        first_col = df.columns[0]
        try:
            dt = pd.to_datetime(df[first_col])
            df.index = dt
            df = df.drop(columns=[first_col])
        except Exception:
            pass
    df = df.sort_index()
    return df


# ========== 指标与回撤 ==========

@dataclass
class DrawdownInfo:
    drawdown: float
    start_date: Optional[pd.Timestamp]
    end_date: Optional[pd.Timestamp]
    recovery_periods: Optional[int]  # 从谷底恢复到峰值所需周数；若未恢复则 None


def compute_max_drawdown(nav: pd.Series) -> DrawdownInfo:
    """
    nav: 累积净值（>0），index 为日期（升序）
    返回最大回撤幅度（正数）、起始/结束日期，以及“回撤区间的天数”（end_date - start_date）
    注意：这里的 recovery_periods 字段将用于存放天数（命名保持兼容，但语义为天数）。
    """
    if nav is None or len(nav) == 0:
        return DrawdownInfo(0.0, None, None, None)

    nav = nav.copy().astype(float)
    nav = nav.replace([np.inf, -np.inf], np.nan).dropna()
    if nav.empty:
        return DrawdownInfo(0.0, None, None, None)

    # 历史峰值轨迹
    peak = nav.cummax()
    # 回撤（正数口径：1 - NAV/Peak）
    dd_pos = 1.0 - (nav / peak)
    # 谷底（最大回撤处）
    t_trough = dd_pos.idxmax()
    max_dd = float(dd_pos.loc[t_trough])  # 已是正数

    # 对应的峰值
    peak_val = float(peak.loc[t_trough])
    # 峰值起点：在 t_trough 之前（含当天）首次达到 peak_val 的日期
    hist = nav.loc[:t_trough]
    eps = 1e-12
    mask_at_peak = (np.abs(hist.values - peak_val) <= eps)
    if not mask_at_peak.any():
        p_idx = int(np.argmax(hist.values))
    else:
        p_idx = int(np.where(mask_at_peak)[0][0])
    t_peak = hist.index[p_idx]

    # 你要求的“天数”：end_date - start_date 的自然日差
    recovery_days = None
    if t_peak is not None and t_trough is not None:
        try:
            recovery_days = int((pd.to_datetime(t_trough) - pd.to_datetime(t_peak)).days)
        except Exception:
            recovery_days = None

    return DrawdownInfo(
        drawdown=max_dd,
        start_date=t_peak,
        end_date=t_trough,
        recovery_periods=recovery_days  # 注意：此处字段名保持不变，但表示天数
    )


def calc_stats_from_returns(rets: np.ndarray,
                            freq_per_year: int = 52,
                            rf_annual: float = 0.0) -> Dict[str, float]:
    """
    使用周收益序列计算指标。
    rf_annual: 年化无风险收益率，换算为每期 rf = rf_annual / freq_per_year
    返回：dict 包含 total_return, annual_return, annual_vol, sharpe, max_drawdown, calmar, n_periods
    """
    rets = np.asarray(rets, dtype=float)
    rets = rets[np.isfinite(rets)]
    n = len(rets)
    if n == 0:
        return {
            "total_return": 0.0, "annual_return": 0.0, "annual_vol": 0.0,
            "sharpe": float("nan"), "max_drawdown": 0.0, "calmar": float("nan"),
            "n_periods": 0
        }
    nav = np.cumprod(1.0 + rets)
    total_return = float(nav[-1] - 1.0)
    ann_return = float((1.0 + total_return) ** (freq_per_year / max(1, n)) - 1.0)
    vol = float(np.std(rets, ddof=1)) * math.sqrt(freq_per_year) if n > 1 else 0.0
    rf_per_period = rf_annual / float(freq_per_year)
    ex = rets - rf_per_period
    sharpe = float(np.mean(ex) / (np.std(ex, ddof=1) if n > 1 else np.nan)) * math.sqrt(freq_per_year)
    # 回撤
    dd_info = compute_max_drawdown(pd.Series(nav))
    calmar = float(ann_return / dd_info.drawdown) if dd_info.drawdown > 1e-12 else float("nan")
    return {
        "total_return": total_return,
        "annual_return": ann_return,
        "annual_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": dd_info.drawdown,
        "calmar": calmar,
        "n_periods": int(n)
    }


def calc_full_metrics(nav_df: pd.DataFrame,
                      ret_col: str = "ret_total",
                      freq_per_year: int = 52,
                      rf_annual: float = 0.0) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], DrawdownInfo]:
    """
    计算总指标 + 逐年指标 + 最大回撤详情
    返回：总指标 dict、年度指标 dict[year -> metrics]、回撤详情 DrawdownInfo
    """
    if nav_df.empty or ret_col not in nav_df.columns:
        return {}, {}, DrawdownInfo(0.0, None, None, None)
    df = nav_df.copy()
    df = df.sort_index()
    rets = df[ret_col].values.astype(float)
    # 总指标
    total_metrics = calc_stats_from_returns(rets, freq_per_year=freq_per_year, rf_annual=rf_annual)
    # 回撤详情基于 NAV
    dd_info = compute_max_drawdown(df["nav"])
    # 年度指标
    by_year: Dict[str, Dict[str, float]] = {}
    df["year"] = df.index.year
    for y, sub in df.groupby("year"):
        m = calc_stats_from_returns(sub[ret_col].values.astype(float), freq_per_year=freq_per_year, rf_annual=rf_annual)
        by_year[str(int(y))] = m
    return total_metrics, by_year, dd_info


# ========== 窗口跨度恢复（用于色块） ==========

def recover_window_spans_from_models(model_dir: Path,
                                     fridays_all: pd.DatetimeIndex,
                                     bt_start: pd.Timestamp,
                                     bt_end: pd.Timestamp,
                                     step_weeks: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    从模型目录里的 model_best_YYYYMMDD.pth + step_weeks 恢复窗口跨度
    返回合并重叠后的 [(start, end)]
    """
    best_files = sorted(model_dir.glob("model_best_*.pth"))
    start_dates = []
    for f in best_files:
        try:
            tag = f.stem.split("_")[-1]
            dt = pd.to_datetime(tag)
            # 对齐至不超过 dt 的最近周五
            if dt not in fridays_all:
                ok = fridays_all[fridays_all <= dt]
                if len(ok) == 0:
                    continue
                dt = ok[-1]
            start_dates.append(dt)
        except Exception:
            continue
    start_dates = sorted(set(start_dates))
    if not start_dates:
        return []
    spans = []
    for s in start_dates:
        idx = fridays_all.get_indexer([s])[0]
        j = min(idx + step_weeks - 1, len(fridays_all) - 1)
        e = fridays_all[j]
        ss = max(s, bt_start)
        ee = min(e, bt_end)
        if ss <= ee:
            spans.append((ss, ee))
    # 合并重叠
    spans_sorted = sorted(spans, key=lambda x: x[0])
    merged: List[List[pd.Timestamp]] = []
    for s, e in spans_sorted:
        if not merged:
            merged.append([s, e])
        else:
            ps, pe = merged[-1]
            if s <= pe:
                merged[-1][1] = max(pe, e)
            else:
                merged.append([s, e])
    return [(s, e) for s, e in merged]


# ========== 绘图 ==========

def plot_nav_compare(
    series_dict: Dict[str, pd.Series],
    out_png: Path,
    title: str,
    spans: Optional[List[Tuple[pd.Timestamp, pd.Timestamp]]] = None,
    excess_series: Optional[Tuple[str, pd.Series]] = None
):
    """
    series_dict: 名称 -> NAV 序列（index=date, value=nav），至少包含主策略
    spans: 可选窗口色块 [(start,end)]
    excess_series: 可选 ("Excess vs XXX", series)
    """
    ensure_dir(out_png)
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    # NAV 曲线
    for name, ser in series_dict.items():
        ax.plot(ser.index, ser.values, label=name, linewidth=1.6)
    # 超额曲线（绘制在同图）
    if excess_series is not None:
        ex_name, ex_ser = excess_series
        ax.plot(ex_ser.index, ex_ser.values, label=ex_name, linewidth=1.8, linestyle="--")
    # 色块
    if spans:
        colors = ["#FFEDA0", "#AEDFF7", "#C7F2C8", "#FBC4AB", "#D9D7F1", "#D7F2BA", "#FDE68A", "#CFE8F3"]
        for i, (s, e) in enumerate(spans):
            c = colors[i % len(colors)]
            ax.axvspan(s, e, color=c, alpha=0.25, zorder=0)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("NAV")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


# ========== 对齐与超额 NAV ==========

def align_nav_frames(dfs: Sequence[pd.DataFrame],
                     how: str = "intersection") -> List[pd.DataFrame]:
    """
    对多个 nav df 进行日期对齐：
    - intersection: 取交集日期
    - union: 取并集，缺失处前向填充（首个非空前仍为空将被丢弃）
    """
    if not dfs:
        return []
    dfs = [df.copy().sort_index() for df in dfs]
    for df in dfs:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
    if how == "union":
        idx = dfs[0].index
        for df in dfs[1:]:
            idx = idx.union(df.index)
        aligned = []
        for df in dfs:
            d = df.reindex(idx).copy()
            if "nav" in d.columns:
                d["nav"] = d["nav"].ffill()
            if "ret_total" in d.columns:
                d["ret_total"] = d["ret_total"].fillna(0.0)
            d = d.dropna(subset=["nav"], how="any")
            aligned.append(d)
        return aligned
    else:
        idx = dfs[0].index
        for df in dfs[1:]:
            idx = idx.intersection(df.index)
        return [df.reindex(idx) for df in dfs]


def excess_nav_from_returns(ret_main: pd.Series,
                            ret_bench: pd.Series) -> pd.Series:
    """
    使用收益差复利生成超额 NAV：
      excess_nav_t = Π (1 + (ret_main_t - ret_bench_t))
    """
    idx = ret_main.index.intersection(ret_bench.index)
    r = (ret_main.loc[idx].values.astype(float) - ret_bench.loc[idx].values.astype(float))
    nav = np.cumprod(1.0 + r)
    return pd.Series(nav, index=idx)


# ========== 年度指标导出 ==========

def metrics_by_year_to_df(metrics_by_year: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    if not metrics_by_year:
        return pd.DataFrame()
    rows = []
    for y, m in metrics_by_year.items():
        row = {"year": y}
        row.update(m)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("year")
    return df