# coding: utf-8
import numpy as np, pandas as pd, pickle, torch
from pathlib import Path
import h5py
from typing import List, Tuple, Optional, Set, Dict

def load_industry_map(csv_file: Path):
    # 保留原函数（兼容旧使用场景）：stock -> 单一 industry id
    df = pd.read_csv(csv_file, encoding='gbk')
    cols = {c.strip().lower(): c for c in df.columns}
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
    df = df.dropna(subset=[key_col, ind_col]).drop_duplicates(subset=[key_col])

    unique_inds = sorted(df[ind_col].unique())
    ind2id = {name: i for i, name in enumerate(unique_inds)}
    stock2id = {stk: ind2id[ind_name] for stk, ind_name in zip(df[key_col], df[ind_col])}
    return stock2id

def load_industry_twolevel_map(csv_file: Path) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    读取 stock_industry_map.csv -> 两级行业映射：
    - 返回 (stock_to_ind1_id, stock_to_ind2_id)
    """
    df = pd.read_csv(csv_file, encoding='gbk')
    cols = {c.strip().lower(): c for c in df.columns}
    key_col = None
    ind1_col = None
    ind2_col = None
    for cand in ['order_book_id', 'code', 'stock_code', 'ticker']:
        if cand in cols:
            key_col = cols[cand]; break
    for cand in ['industry', 'first_industry_name', 'industry_name', 'industry_cn']:
        if cand in cols:
            ind1_col = cols[cand]; break
    for cand in ['industry2', 'second_industry_name', 'industry_name_2', 'industry2_cn']:
        if cand in cols:
            ind2_col = cols[cand]; break
    if key_col is None or ind1_col is None:
        raise ValueError(f"行业映射缺少列，检测到: {df.columns.tolist()}，至少需 order_book_id + industry；建议包含 industry2")
    df = df[[key_col, ind1_col] + ([ind2_col] if ind2_col else [])].copy()
    df[key_col]  = df[key_col].astype(str).str.strip()
    df[ind1_col] = df[ind1_col].astype(str).str.strip()
    if ind2_col:
        df[ind2_col] = df[ind2_col].astype(str).str.strip()

    # 构造 id
    uniq1 = sorted(df[ind1_col].dropna().unique())
    id1_map = {v: i for i, v in enumerate(uniq1)}
    if ind2_col:
        uniq2 = sorted(df[ind2_col].dropna().unique())
        id2_map = {v: i for i, v in enumerate(uniq2)}
    else:
        id2_map = {}

    stock2ind1 = {r[key_col]: id1_map[r[ind1_col]] for _, r in df.dropna(subset=[ind1_col]).iterrows()}
    if ind2_col:
        stock2ind2 = {r[key_col]: id2_map[r[ind2_col]] for _, r in df.dropna(subset=[ind2_col]).iterrows()}
    else:
        stock2ind2 = {}
    return stock2ind1, stock2ind2

def load_chain_sector_map(csv_file: Path) -> Dict[str, int]:
    """
    读取 stock_style_map.csv -> chain_sector 映射：stock -> chain_id
    需要列：order_book_id, chain_sector
    """
    df = pd.read_csv(csv_file, encoding='gbk')
    cols = {c.strip().lower(): c for c in df.columns}
    key_col = None
    chain_col = None
    for cand in ['order_book_id', 'code', 'stock_code', 'ticker']:
        if cand in cols:
            key_col = cols[cand]; break
    for cand in ['chain_sector', 'industry_chain_sector_name', 'chain', 'chain_name']:
        if cand in cols:
            chain_col = cols[cand]; break
    if key_col is None or chain_col is None:
        raise ValueError(f"风格链映射缺少列，检测到: {df.columns.tolist()}，需要 order_book_id + chain_sector")

    df = df[[key_col, chain_col]].copy()
    df[key_col]   = df[key_col].astype(str).str.strip()
    df[chain_col] = df[chain_col].astype(str).str.strip()
    df = df.dropna(subset=[key_col, chain_col]).drop_duplicates(subset=[key_col])

    uniq = sorted(df[chain_col].unique())
    c2id = {v: i for i, v in enumerate(uniq)}
    return {r[key_col]: c2id[r[chain_col]] for _, r in df.iterrows()}

# -------- 交易日历 --------
def load_calendar(csv_file: Path) -> pd.DatetimeIndex:
    dates = pd.read_csv(csv_file, header=None, skiprows=1).iloc[:, 1]
    return pd.DatetimeIndex(sorted(pd.to_datetime(dates, format='%Y-%m-%d').unique()))

def get_next_trading_day(date_str,csv_file: Path):
    trading_days = load_calendar(csv_file)
    input_date = pd.to_datetime(date_str)
    current_index = trading_days.get_loc(input_date)
    return trading_days[current_index + 1].strftime('%Y-%m-%d')

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
    n = pred.numel()
    if n < 2:
        return 0.0
    pr = pred.argsort().argsort().float()
    tg = tgt.argsort().argsort().float()
    pr_std = pr.std(unbiased=False)
    tg_std = tg.std(unbiased=False)
    if pr_std <= 1e-12 or tg_std <= 1e-12:
        return 0.0
    pr = (pr - pr.mean()) / (pr_std + 1e-8)
    tg = (tg - tg.mean()) / (tg_std + 1e-8)
    return (pr * tg).mean().item()

# ========== 新增：增量特征构建工具 ==========
def read_h5_meta(h5_path: Path) -> Tuple[int, Set[pd.Timestamp], Optional[List[str]]]:
    if not h5_path.exists():
        return 0, set(), None
    written_dates = set()
    factor_cols = None
    with h5py.File(h5_path, "r") as h5f:
        if 'factor_cols' in h5f.attrs:
            factor_cols = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in np.array(h5f.attrs['factor_cols'])]
        for k in h5f.keys():
            if not k.startswith("date_"):
                continue
            d_str = h5f[k].attrs.get('date', None)
            if d_str is None:
                continue
            if isinstance(d_str, bytes):
                d_str = d_str.decode('utf-8')
            written_dates.add(pd.to_datetime(d_str))
        next_idx = sum(1 for k in h5f.keys() if k.startswith("date_"))
    return next_idx, written_dates, factor_cols

def list_missing_fridays(calendar_csv: Path, start_date: str, end_date: str, written_dates: Set[pd.Timestamp]) -> pd.DatetimeIndex:
    cal = load_calendar(calendar_csv)
    all_fridays = weekly_fridays(cal)
    mask = (all_fridays >= pd.Timestamp(start_date)) & (all_fridays <= pd.Timestamp(end_date))
    all_fridays = all_fridays[mask]
    need = [d for d in all_fridays if d not in written_dates]
    return pd.DatetimeIndex(sorted(need))

def get_required_history_start(missing_fridays: pd.DatetimeIndex, max_lookback: int) -> Optional[pd.Timestamp]:
    if missing_fridays is None or len(missing_fridays) == 0:
        return None
    first_target = missing_fridays.min()
    start_date = first_target - pd.Timedelta(days=max_lookback * 2)
    return start_date.normalize()

def ensure_multiindex_price(day_df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(day_df.index, pd.MultiIndex):
        raise ValueError("price_day_file 需要为 MultiIndex(order_book_id, datetime) 格式。")
    names = list(day_df.index.names)
    if names != ['order_book_id', 'datetime']:
        try:
            day_df.index = day_df.index.set_names(['order_book_id', 'datetime'])
        except Exception:
            pass
    return day_df.sort_index()

def merge_multiindex_columns_union(df_old: pd.DataFrame, df_new: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    cols = sorted(set(df_old.columns) | set(df_new.columns))
    return df_old.reindex(columns=cols), df_new.reindex(columns=cols), cols

class SlidingWindowCache:
    def __init__(self): 
        self.cache: Dict[str, Tuple[pd.Timestamp, pd.DataFrame]] = {}
    def get(self, code: str, end_dt: pd.Timestamp, win: int) -> Optional[pd.DataFrame]:
        if code not in self.cache: return None
        last_end, last_df = self.cache[code]
        if last_df is None or last_end is None: return None
        if end_dt >= last_end and len(last_df) >= win:
            return last_df.tail(win)
        return None
    def put(self, code: str, end_dt: pd.Timestamp, df: pd.DataFrame):
        self.cache[code] = (end_dt, df.copy())