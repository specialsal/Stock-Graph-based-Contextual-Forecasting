# coding: utf-8
"""
工具函数
"""
import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Tuple
from pathlib import Path


def load_trading_calendar(calendar_file: Path) -> pd.DatetimeIndex:
    """加载交易日历"""
    # 跳过第一行（,0），从第二行开始读取
    df = pd.read_csv(calendar_file, header=None, skiprows=1)

    # 取第二列（索引1）作为日期数据
    dates = pd.to_datetime(df.iloc[:, 1], format='%Y/%m/%d')

    # 修复：直接对DatetimeIndex进行排序
    return pd.DatetimeIndex(sorted(dates.unique()))


def week_last_trading_days(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """返回每周最后一个交易日"""
    df = pd.DataFrame({'date': calendar})
    df['week'] = df['date'].dt.to_period('W')
    weekly_last = df.groupby('week')['date'].last()
    return pd.DatetimeIndex(weekly_last.values)


def mad_clip(arr: np.ndarray, k: float = 3.0) -> np.ndarray:
    """MAD去极值"""
    median = np.nanmedian(arr, axis=0, keepdims=True)
    mad = np.nanmedian(np.abs(arr - median), axis=0, keepdims=True) + 1e-6
    upper = median + k * mad
    lower = median - k * mad
    return np.clip(arr, lower, upper)


class Scaler:
    """标准化器"""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x: np.ndarray):
        """计算均值和标准差"""
        self.mean = np.nanmean(x, axis=(0, 1), keepdims=True)
        self.std = np.nanstd(x, axis=(0, 1), keepdims=True) + 1e-6

    def transform(self, x: np.ndarray) -> np.ndarray:
        """标准化"""
        return (x - self.mean) / self.std

    def save(self, path: Path):
        """保存参数"""
        with open(path, 'wb') as f:
            pickle.dump({'mean': self.mean, 'std': self.std}, f)

    def load(self, path: Path):
        """加载参数"""
        with open(path, 'rb') as f:
            params = pickle.load(f)
            self.mean = params['mean']
            self.std = params['std']


def save_universe(universe: Dict, path: Path):
    """保存股票池"""
    with open(path, 'wb') as f:
        pickle.dump(universe, f)


def load_universe(path: Path) -> Dict:
    """加载股票池"""
    with open(path, 'rb') as f:
        return pickle.load(f)