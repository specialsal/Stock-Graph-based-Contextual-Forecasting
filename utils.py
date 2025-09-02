# coding: utf-8
import numpy as np, pandas as pd, pickle, torch
from pathlib import Path

def load_industry_map(csv_file: Path):
    # 使用 gbk 编码，兼容列名的潜在空白与大小写
    df = pd.read_csv(csv_file, encoding='gbk')
    cols = {c.strip().lower(): c for c in df.columns}
    # 兼容可能的列名变体
    key_col = None
    ind_col = None
    for cand in ['order_book_id', 'code', 'stock_code', 'ticker']:
        if cand in cols:
            key_col = cols[cand]
            break
    for cand in ['industry', 'industry_name', 'sector', 'industry_cn']:
        if cand in cols:
            ind_col = cols[cand]
            break
    if key_col is None or ind_col is None:
        raise ValueError(f"行业映射CSV缺少必要列，检测到列: {df.columns.tolist()}，需要至少包含 order_book_id 与 industry")

    df = df[[key_col, ind_col]].copy()
    df[key_col] = df[key_col].astype(str).str.strip()
    df[ind_col] = df[ind_col].astype(str).str.strip()

    # 去重，保留首次出现
    df = df.dropna(subset=[key_col, ind_col]).drop_duplicates(subset=[key_col])

    # 将中文行业名映射为稳定整数ID（按名称排序，保证可复现）
    unique_inds = sorted(df[ind_col].unique())
    ind2id = {name: i for i, name in enumerate(unique_inds)}  # 0..K-1
    stock2id = {stk: ind2id[ind_name] for stk, ind_name in zip(df[key_col], df[ind_col])}

    # 返回股票->行业ID 映射；如需行业名->ID也可返回
    return stock2id  # 与现有调用兼容
# -------- 交易日历 --------
def load_calendar(csv_file: Path) -> pd.DatetimeIndex:
    dates = pd.read_csv(csv_file, header=None, skiprows=1).iloc[:, 1]
    return pd.DatetimeIndex(sorted(pd.to_datetime(dates, format='%Y/%m/%d').unique()))

def weekly_fridays(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    df = pd.DataFrame({"d": calendar})
    df['w'] = df['d'].dt.to_period('W')
    return pd.DatetimeIndex(df.groupby('w')['d'].last().values)

# -------- MAD 裁剪 + z-score --------
def mad_clip(x: np.ndarray, k: float = 3.0):
    med = np.nanmedian(x, 0, keepdims=True)
    mad = np.nanmedian(np.abs(x - med), 0, keepdims=True) + 1e-6
    return np.clip(x, med - k * mad, med + k * mad)

class Scaler:
    def __init__(self): self.mean = None; self.std = None
    def fit(self, arr): 
        self.mean = np.nanmean(arr, (0, 1), keepdims=True)
        self.std  = np.nanstd (arr, (0, 1), keepdims=True) + 1e-6
    def transform(self, arr): return (arr - self.mean) / self.std

# -------- 保存 / 读取任意对象 --------
def pkl_dump(obj, path: Path):
    with open(path, "wb") as f: pickle.dump(obj, f)

def pkl_load(path: Path):
    with open(path, "rb") as f: return pickle.load(f)

# -------- RankIC（torch，无梯度）--------
@torch.no_grad()
def rank_ic(pred, tgt):
    # pred, tgt: 1D tensor
    n = pred.numel()
    if n < 2:
        return 0.0
    pr = pred.argsort().argsort().float()
    tg = tgt.argsort().argsort().float()
    # z-score 前检查方差
    pr_std = pr.std(unbiased=False)
    tg_std = tg.std(unbiased=False)
    if pr_std <= 1e-12 or tg_std <= 1e-12:
        return 0.0
    pr = (pr - pr.mean()) / (pr_std + 1e-8)
    tg = (tg - tg.mean()) / (tg_std + 1e-8)
    return (pr * tg).mean().item()