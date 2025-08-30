# coding: utf-8
"""
Quant ML Pipeline
-----------------
日频 + 30 分钟 + 行业环境端到端模型
适用于周度截面选股
"""

import os
import math
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

########################
# 1. 配置
########################
@dataclass
class Config:
    # 数据路径
    root = "./data"                       # 根目录
    price_daily = "daily.parquet"         # 日频 OHLCV
    price_min = "min30.parquet"           # 30 分钟 OHLCV
    env_file = "industry_env.parquet"     # 行业轮动指标
    meta_file = "meta.parquet"            # IPO、退市、ST、停牌
    industry_map = "industry_map.csv"     # stock -> industry
    trading_calendar = "trading_calendar.csv"  # 交易日历文件

    # 时间参数
    daily_window = 120                    # 日线窗口长度
    min_window = 80                       # 30min 窗口长度 (10d * 8)
    env_window = 150                      # 行业序列窗口
    sample_freq = "W-FRI"                 # 每周五采样

    # 过滤规则
    ipo_cut = 160                         # 上市≤160日的新股剔除
    st_exclude = True                     # 是否剔除 ST
    suspended_exclude = True              # 是否剔除停牌
    min_daily_volume = 1e6                # 最小成交额过滤

    # 训练参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden = 128
    batch_size = 1024
    lr = 3e-4
    weight_decay = 1e-2
    alpha = 0.7                           # IC Loss 权重
    epochs = 20

CFG = Config()
########################
# 2. 工具函数
########################


def load_trading_calendar(calendar_file: str) -> pd.DatetimeIndex:
    """
    从CSV文件读取交易日历

    Parameters:
    - calendar_file: 交易日历CSV文件路径

    Returns:
    - calendar: 交易日历DatetimeIndex
    """
    # 读取CSV文件，假设日期在B列（第二列）
    calendar_df = pd.read_csv(calendar_file)

    # 如果日期在B列，使用iloc[:,1]；如果有列名可以直接用列名
    # 这里假设日期列名为第二列，你可以根据实际情况调整
    if calendar_df.shape[1] > 1:
        date_col = calendar_df.iloc[:, 1]  # B列
    else:
        date_col = calendar_df.iloc[:, 0]  # 如果只有一列

    # 转换为日期格式
    calendar = pd.to_datetime(date_col, format='%Y/%m/%d')

    # 去重并排序
    calendar = calendar.drop_duplicates().sort_values()

    return pd.DatetimeIndex(calendar)


def generate_weekly_labels(daily_df: pd.DataFrame, sample_dates: pd.DatetimeIndex,
                           calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """
    生成真正的下一周收益率标签
    """
    close_pivot = daily_df["close"].unstack()  # 转换为 date x stock 格式

    # 构建每个采样日对应的下周交易日区间
    weekly_returns = {}

    for i, current_friday in enumerate(sample_dates):
        if i == len(sample_dates) - 1:
            # 最后一个周五，无法计算下周收益
            continue

        next_friday = sample_dates[i + 1]

        # 找到当前周五到下一个周五之间的所有交易日
        next_week_mask = (calendar > current_friday) & (calendar <= next_friday)
        next_week_trading_days = calendar[next_week_mask]

        if len(next_week_trading_days) == 0:
            continue

        # 确保开始和结束日期都在数据中
        if current_friday not in close_pivot.index or next_week_trading_days[-1] not in close_pivot.index:
            continue

        # 计算下一周的收益率
        start_price = close_pivot.loc[current_friday]  # 当前周五收盘价
        end_price = close_pivot.loc[next_week_trading_days[-1]]  # 下周最后一个交易日收盘价

        weekly_return = (end_price / start_price) - 1
        weekly_returns[current_friday] = weekly_return

    # 转换为 MultiIndex DataFrame
    label_list = []
    for date, returns in weekly_returns.items():
        for stock in returns.index:
            if not pd.isna(returns[stock]):
                label_list.append({
                    'date': date,
                    'stock': stock,
                    'next_week_return': returns[stock]
                })

    label_df = pd.DataFrame(label_list)
    if len(label_df) > 0:
        label_df = label_df.set_index(['date', 'stock'])['next_week_return']
    else:
        # 如果没有数据，创建空的MultiIndex DataFrame
        label_df = pd.DataFrame(index=pd.MultiIndex.from_tuples([], names=['date', 'stock']),
                                columns=['next_week_return'])['next_week_return']

    return label_df

def load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

def week_last_trading_days(cal: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """返回每周最后一个交易日"""
    return cal.to_series().groupby(cal.to_series().dt.isocalendar().week).last().index

########################
# 3. 股票可交易宇宙过滤
########################
def build_universe(meta: pd.DataFrame, dates: pd.DatetimeIndex) -> Dict[pd.Timestamp, List[str]]:
    """
    meta: index=trade_date, columns=[stock, is_st, is_suspended, ipo_date, delist_date, daily_amount]
    输出: {date: [stock1, stock2, ...]}
    """
    universe = {}
    for d in tqdm(dates, desc="Building universe"):
        today = meta.loc[d]               # 当日全部元信息
        cond = (
            (today["days_since_ipo"] >= CFG.ipo_cut) &
            (today["is_st"] == 0 if CFG.st_exclude else True) &
            (today["is_sus"] == 0 if CFG.suspended_exclude else True) &
            (today["daily_amount"] >= CFG.min_daily_volume)
        )
        universe[d] = today[cond].index.tolist()
    return universe

########################
# 4. 特征工程与标准化
########################
class Scaler:
    """滚动保存均值方差；仅在训练集 fit，一次性保存"""
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x: np.ndarray):
        self.mean = x.mean(axis=(0, 1), keepdims=True)
        self.std = x.std(axis=(0, 1), keepdims=True) + 1e-6

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

def mad_clip(arr: np.ndarray, k: float = 3.0) -> np.ndarray:
    """MAD 去极值"""
    median = np.median(arr, axis=0, keepdims=True)
    mad = np.median(np.abs(arr - median), axis=0, keepdims=True) + 1e-6
    upper = median + k * mad
    lower = median - k * mad
    return np.clip(arr, lower, upper)

########################
# 5. Dataset / Dataloader
########################
class StockDataset(Dataset):
    """周度截面 Dataset，每个元素 = 单支股票的多模态窗口"""
    def __init__(
        self,
        dates: List[pd.Timestamp],
        universe: Dict[pd.Timestamp, List[str]],
        daily_df: pd.DataFrame,
        min_df: pd.DataFrame,
        env_df: pd.DataFrame,
        scaler_daily: Scaler,
        scaler_min: Scaler,
        industry_map: pd.Series,
        label_df: pd.DataFrame
    ):
        self.dates = dates
        self.universe = universe
        self.daily = daily_df
        self.min30 = min_df
        self.env = env_df
        self.scaler_d = scaler_daily
        self.scaler_m = scaler_min
        self.industry_map = industry_map
        self.label = label_df

        # 预展开索引 (date, stock) -> 方便 __getitem__
        self.index = []
        for d in dates:
            for s in universe[d]:
                self.index.append((d, s))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        d, s = self.index[idx]

        # 1. 取日频窗口（含 d 当天）
        d_end = d
        d_start = d - pd.Timedelta(days=CFG.daily_window-1)
        x_d = self.daily.loc[(slice(d_start, d_end), s)].values
        x_d = self.scaler_d.transform(mad_clip(x_d.astype(np.float32)))

        # 2. 取 30min 窗口（d 当天盘中 bar + 前 9 日）
        m_end = pd.Timestamp(d.strftime("%Y-%m-%d 15:00:00"))
        m_start = m_end - pd.Timedelta(minutes=30*(CFG.min_window-1))
        x_m = self.min30.loc[(slice(m_start, m_end), s)].values
        x_m = self.scaler_m.transform(mad_clip(x_m.astype(np.float32)))

        # 3. 行业环境窗口（滞后一天）
        e_end = d - pd.Timedelta(days=1)
        e_start = e_end - pd.Timedelta(days=CFG.env_window-1)
        ind = self.industry_map[s]
        x_e = self.env.loc[(slice(e_start, e_end), ind)].values.astype(np.float32)

        # 4. Label: 下一周收益
        y = self.label.loc[(d, s)]

        return {
            "x_d": torch.tensor(x_d, dtype=torch.float32),
            "x_m": torch.tensor(x_m, dtype=torch.float32),
            "x_e": torch.tensor(x_e, dtype=torch.float32),
            "ind_id": torch.tensor(ind, dtype=torch.long),
            "label": torch.tensor(y, dtype=torch.float32)
        }

########################
# 6. 模型
########################
class DailyEncoder(nn.Module):
    """日频 Transformer"""
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, CFG.daily_window, in_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_dim, nhead=4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(in_dim, hidden)

    def forward(self, x):
        x = x + self.pe
        h = self.encoder(x)
        h = h.mean(dim=1)                 # Global average pooling
        return self.fc(h)

class MinuteEncoder(nn.Module):
    """30min Conv + GRU + 注意力"""
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, in_dim, kernel_size=3, stride=2, padding=1)
        self.gru = nn.GRU(in_dim, hidden, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden*2, 1)

    def forward(self, x):
        # x: [B,T,C] -> Conv1d 需要 [B,C,T]
        x = x.transpose(1,2)
        x = self.conv(x).transpose(1,2)
        out,_ = self.gru(x)
        w = torch.softmax(self.attn(out), dim=1)
        h = (out * w).sum(dim=1)
        return h                           # [B, hidden*2]

class EnvEncoder(nn.Module):
    """行业环境 Transformer"""
    def __init__(self, in_dim, hidden, n_ind):
        super().__init__()
        self.ind_emb = nn.Embedding(n_ind, 32)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_dim, nhead=4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(in_dim+32, hidden)

    def forward(self, x, ind_id):
        h = self.encoder(x).mean(dim=1)    # [B, in_dim]
        ind_vec = self.ind_emb(ind_id)
        return self.fc(torch.cat([h, ind_vec], dim=-1))

class FusionModel(nn.Module):
    def __init__(self, dim_d, dim_m, dim_e, n_ind):
        super().__init__()
        self.enc_d = DailyEncoder(dim_d, CFG.hidden)
        self.enc_m = MinuteEncoder(dim_m, CFG.hidden)
        self.enc_e = EnvEncoder(dim_e, CFG.hidden, n_ind)
        self.cross = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=CFG.hidden*3, nhead=8, batch_first=True), 1)
        self.mlp = nn.Sequential(
            nn.LayerNorm(CFG.hidden*3),
            nn.Linear(CFG.hidden*3, 256),
            nn.SiLU(),
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x_d, x_m, x_e, ind_id):
        h_d = self.enc_d(x_d)
        h_m = self.enc_m(x_m)
        h_e = self.enc_e(x_e, ind_id)
        h = torch.cat([h_d, h_m, h_e], dim=-1).unsqueeze(1)   # [B,1,H*3]
        h = self.cross(h).squeeze(1)
        return self.mlp(h).squeeze(-1)        # [B]

########################
# 7. RankIC & Loss
########################
def rank_ic(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """计算截面 RankIC (Spearman)"""
    pred_rank = pred.argsort().argsort().float()
    target_rank = target.argsort().argsort().float()
    cov = torch.dot(pred_rank - pred_rank.mean(), target_rank - target_rank.mean())
    ic = cov / (pred_rank.std() * target_rank.std() + 1e-6)
    return ic

def ic_loss(pred, target):
    return 1 - rank_ic(pred, target)

def pairwise_hinge(pred, target, margin=0.0):
    """同截面股票两两配对"""
    diff_pred = pred.unsqueeze(0) - pred.unsqueeze(1)     # P_i - P_j
    diff_true = target.unsqueeze(0) - target.unsqueeze(1) # R_i - R_j
    label = torch.sign(diff_true)
    loss = torch.relu(margin - label * diff_pred)
    return loss.mean()

########################
# 8. 训练 & 验证循环
########################
def train_one_epoch(model, loader, opt, scaler_loss=True):
    model.train()
    total_loss, total_ic = 0., 0.
    for batch in loader:
        for k in batch:
            batch[k] = batch[k].to(CFG.device)
        pred = model(batch["x_d"], batch["x_m"], batch["x_e"], batch["ind_id"])
        l_ic = ic_loss(pred, batch["label"])
        l_pair = pairwise_hinge(pred, batch["label"])
        loss = CFG.alpha * l_ic + (1-CFG.alpha) * l_pair

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        total_loss += loss.item()*len(pred)
        total_ic += rank_ic(pred.detach(), batch["label"]).item()*len(pred)

    n = len(loader.dataset)
    return total_loss/n, total_ic/n

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_ic = 0.
    for batch in loader:
        for k in batch:
            batch[k] = batch[k].to(CFG.device)
        pred = model(batch["x_d"], batch["x_m"], batch["x_e"], batch["ind_id"])
        total_ic += rank_ic(pred, batch["label"]).item()*len(pred)
    return total_ic / len(loader.dataset)

########################
# 9. 主流程 (伪代码骨架)
########################
def main():
    # 1) 读取数据
    daily = load_parquet(os.path.join(CFG.root, CFG.price_daily))
    min30 = load_parquet(os.path.join(CFG.root, CFG.price_min))
    env = load_parquet(os.path.join(CFG.root, CFG.env_file))
    meta = load_parquet(os.path.join(CFG.root, CFG.meta_file))
    ind_map = pd.read_csv(os.path.join(CFG.root, CFG.industry_map), index_col=0, squeeze=True)

    # 2) 读取交易日历并构造采样日
    calendar = load_trading_calendar(os.path.join(CFG.root, CFG.trading_calendar))
    sample_dates = week_last_trading_days(calendar)

    # 3) 计算元信息衍生列
    meta["days_since_ipo"] = (meta.index.get_level_values(0) - meta["ipo_date"]).dt.days
    meta = meta.rename(columns={"is_suspended": "is_sus"})

    # 4) Universe
    universe = build_universe(meta, sample_dates)

    # 5) 生成标签(下一周收益) - 使用真实交易日历
    label_df = generate_weekly_labels(daily, sample_dates, calendar)

    # 6) 划分 Train/Val/Test 日期
    train_dates = [d for d in sample_dates if d.year <= 2021]
    val_dates = [d for d in sample_dates if d.year == 2022]
    test_dates = [d for d in sample_dates if d.year >= 2023]

    # 7) 统计量拟合
    scaler_d, scaler_m = Scaler(), Scaler()

    # 只在训练集股票上 fit
    train_daily_samples = []
    for d in train_dates:
        stocks = universe[d]
        d_end = d
        d_start = d - pd.Timedelta(days=CFG.daily_window-1)
        arr = daily.loc[(slice(d_start, d_end), stocks)].values
        train_daily_samples.append(arr)
    scaler_d.fit(np.concatenate(train_daily_samples, axis=0))

    train_min_samples = []
    for d in train_dates:
        stocks = universe[d]
        m_end = pd.Timestamp(d.strftime("%Y-%m-%d 15:00:00"))
        m_start = m_end - pd.Timedelta(minutes=30*(CFG.min_window-1))
        arr = min30.loc[(slice(m_start, m_end), stocks)].values
        train_min_samples.append(arr)
    scaler_m.fit(np.concatenate(train_min_samples, axis=0))

    # 8) 构造 Dataset / Loader
    train_set = StockDataset(train_dates, universe, daily, min30, env,
                             scaler_d, scaler_m, ind_map, label_df)
    val_set = StockDataset(val_dates, universe, daily, min30, env,
                           scaler_d, scaler_m, ind_map, label_df)

    train_loader = DataLoader(train_set, batch_size=CFG.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=CFG.batch_size, shuffle=False, num_workers=4)

    # 9) 模型 / 优化器
    dim_d = daily.shape[1] // len(daily.columns.levels[1])      # 日频字段数
    dim_m = min30.shape[1] // len(min30.columns.levels[1])
    dim_e = env.shape[1] // len(env.columns.levels[1])
    n_ind = ind_map.nunique()

    model = FusionModel(dim_d, dim_m, dim_e, n_ind).to(CFG.device)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)

    # 10) 训练
    for epoch in range(1, CFG.epochs+1):
        tr_loss, tr_ic = train_one_epoch(model, train_loader, opt)
        val_ic = evaluate(model, val_loader)
        print(f"Epoch {epoch:02d}  Train Loss {tr_loss:.4f}  Train IC {tr_ic:.4f}  Val IC {val_ic:.4f}")

    # 11) 推理示例 (test_dates)
    model.eval()
    test_set = StockDataset(test_dates, universe, daily, min30, env,
                            scaler_d, scaler_m, ind_map, label_df)
    test_loader = DataLoader(test_set, batch_size=CFG.batch_size, shuffle=False)

    preds, rets = [], []
    for batch in test_loader:
        for k in batch:
            batch[k] = batch[k].to(CFG.device)
        with torch.no_grad():
            p = model(batch["x_d"], batch["x_m"], batch["x_e"], batch["ind_id"])
        preds.append(p.cpu())
        rets.append(batch["label"].cpu())
    preds = torch.cat(preds); rets = torch.cat(rets)
    print("Test RankIC:", rank_ic(preds, rets).item())

if __name__ == "__main__":
    main()