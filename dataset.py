# coding: utf-8
"""
PyTorch Dataset类
"""
import h5py
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List
from utils import mad_clip, load_universe


class StockDataset(Dataset):
    """股票数据集"""

    def __init__(
            self,
            features_path: Path,
            label_path: Path,
            scaler_path: Path,
            universe_path: Path
    ):
        """
        Parameters:
        -----------
        features_path: 特征文件路径 (h5)
        label_path: 标签文件路径 (parquet)
        scaler_path: 标准化器路径 (pkl)
        universe_path: 股票池路径 (pkl)
        """
        self.features_path = features_path

        # 加载标签
        self.labels = pd.read_parquet(label_path)

        # 加载标准化器
        with open(scaler_path, 'rb') as f:
            scalers = pickle.load(f)
            self.daily_scaler = scalers['daily']
            self.min30_scaler = scalers['min30']

        # 加载股票池
        self.universe = load_universe(universe_path)

        # 构建索引
        self._build_index()

    def _build_index(self):
        """构建数据索引"""
        self.index = []

        with h5py.File(self.features_path, 'r') as h5f:
            for date_key in h5f.keys():
                date_str = h5f[date_key].attrs['date']
                date = pd.Timestamp(date_str)
                stocks = h5f[date_key].attrs['stocks'].tolist()

                for stock_idx, stock in enumerate(stocks):
                    # 检查是否有标签
                    if (date, stock) in self.labels.index:
                        self.index.append({
                            'date_key': date_key,
                            'date': date,
                            'stock': stock,
                            'stock_idx': stock_idx
                        })

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        item = self.index[idx]

        with h5py.File(self.features_path, 'r') as h5f:
            date_group = h5f[item['date_key']]

            # 获取日频特征
            daily_features = date_group['daily'][item['stock_idx']]  # [T, C]
            # 处理缺失值和异常值
            if not np.isnan(daily_features).all():
                daily_features = mad_clip(daily_features)
                daily_features = self.daily_scaler.transform(daily_features[np.newaxis, ...])[0]
            else:
                daily_features = np.zeros_like(daily_features)

            # 获取30分钟特征
            min30_features = date_group['min30'][item['stock_idx']]  # [T, C]
            if not np.isnan(min30_features).all():
                min30_features = mad_clip(min30_features)
                min30_features = self.min30_scaler.transform(min30_features[np.newaxis, ...])[0]
            else:
                min30_features = np.zeros_like(min30_features)

        # 获取标签
        label = self.labels.loc[(item['date'], item['stock']), 'next_week_return']

        return {
            'daily': torch.tensor(daily_features, dtype=torch.float32),
            'min30': torch.tensor(min30_features, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.float32),
            'date': item['date'].strftime('%Y-%m-%d'),
            'stock': item['stock']
        }


def collate_fn(batch):
    """自定义collate函数"""
    return {
        'daily': torch.stack([item['daily'] for item in batch]),
        'min30': torch.stack([item['min30'] for item in batch]),
        'label': torch.stack([item['label'] for item in batch]),
        'dates': [item['date'] for item in batch],
        'stocks': [item['stock'] for item in batch]
    }