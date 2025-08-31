# coding: utf-8
"""
通用工具函数
"""
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import torch


# ---------------- 交易日历 ----------------
def load_trading_calendar(calendar_file: Path) -> pd.DatetimeIndex:
    df = pd.read_csv(calendar_file, header=None, skiprows=1)
    dates = pd.to_datetime(df.iloc[:, 1], format='%Y/%m/%d')
    return pd.DatetimeIndex(sorted(dates.unique()))


def week_last_trading_days(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    df = pd.DataFrame({'date': calendar})
    df['week'] = df['date'].dt.to_period('W')
    weekly_last = df.groupby('week')['date'].last()
    return pd.DatetimeIndex(weekly_last.values)


# ---------------- 数据处理 ----------------
def mad_clip(arr: np.ndarray, k: float = 3.0) -> np.ndarray:
    median = np.nanmedian(arr, axis=0, keepdims=True)
    mad = np.nanmedian(np.abs(arr - median), axis=0, keepdims=True) + 1e-6
    upper = median + k * mad
    lower = median - k * mad
    return np.clip(arr, lower, upper)


class Scaler:
    def __init__(self):
        self.mean, self.std = None, None

    def fit(self, x: np.ndarray):
        self.mean = np.nanmean(x, axis=(0, 1), keepdims=True)
        self.std  = np.nanstd(x,  axis=(0, 1), keepdims=True) + 1e-6

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


# ---------------- 股票池 ----------------
def save_universe(obj, path: Path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_universe(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------- 可导排序相关 ----------------
def soft_rank(x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """
    Differentiable ranking approximation (O(N²)).
    论文: Blondel et al. NeurIPS 2020
    """
    n = x.size(-1)
    diff = x.unsqueeze(-1) - x.unsqueeze(-2)        # [..., N, N]
    P = torch.sigmoid(-diff / tau)                  # 蚂蚁概率
    SR = P.sum(dim=-1) + 0.5                        # [..., N]
    return SR


def soft_ic_loss(pred: torch.Tensor,
                 tgt:  torch.Tensor,
                 tau:  float = 1.0) -> torch.Tensor:
    """
    可导 Spearman IC Loss = 1 - Corr(pred, soft_rank(label))
    """
    p = (pred - pred.mean()) / (pred.std() + 1e-8)
    r = soft_rank(tgt, tau=tau)
    r = (r - r.mean()) / (r.std() + 1e-8)
    return 1.0 - (p * r).mean()