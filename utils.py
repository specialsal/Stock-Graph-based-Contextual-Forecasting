# coding: utf-8
import numpy as np, pandas as pd, pickle, torch
from pathlib import Path
import h5py
from typing import List, Tuple, Optional, Set, Dict

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
    return stock2id

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

# ========== 新增：增量特征构建工具 ==========
def read_h5_meta(h5_path: Path) -> Tuple[int, Set[pd.Timestamp], Optional[List[str]]]:
    """
    读取已存在的 H5 特征仓元信息：
    - 返回 下一 group 序号、已写入的周五日期集合、factor_cols（若存在）
    """
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
    """
    基于交易日历计算[start_date, end_date]内的全部周五，并去掉已写入组，得到缺失周五列表
    """
    cal = load_calendar(calendar_csv)
    all_fridays = weekly_fridays(cal)
    mask = (all_fridays >= pd.Timestamp(start_date)) & (all_fridays <= pd.Timestamp(end_date))
    all_fridays = all_fridays[mask]
    need = [d for d in all_fridays if d not in written_dates]
    return pd.DatetimeIndex(sorted(need))

def get_required_history_start(missing_fridays: pd.DatetimeIndex, max_lookback: int) -> Optional[pd.Timestamp]:
    """
    对于需要增量的最早周五，向前回溯 max_lookback-1 个交易日作为计算窗口的起点
    注意：这里用自然日减法，若你需要精确“交易日回溯”，请在调用前基于交易日序列做 index-based 回退。
    """
    if missing_fridays is None or len(missing_fridays) == 0:
        return None
    first_target = missing_fridays.min()
    # 使用自然日减去一定天数，保守取值，多取一点冗余并不影响正确性
    start_date = first_target - pd.Timedelta(days=max_lookback * 2)
    return start_date.normalize()

def ensure_multiindex_price(day_df: pd.DataFrame) -> pd.DataFrame:
    """
    确保日线行情为 MultiIndex(order_book_id, datetime) 并按索引排序
    """
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
    """
    对齐两份列名的并集并返回对齐后的副本与并集列顺序
    """
    cols = sorted(set(df_old.columns) | set(df_new.columns))
    return df_old.reindex(columns=cols), df_new.reindex(columns=cols), cols

class SlidingWindowCache:
    """
    简单的滑窗缓存器：缓存每只股票最近一次切片的（end_date, window_df），
    若下一次请求的 end_date 相同或更晚且窗口相同，则避免重复切 slice。
    """
    def __init__(self): self.cache: Dict[str, Tuple[pd.Timestamp, pd.DataFrame]] = {}
    def get(self, code: str, end_dt: pd.Timestamp, win: int) -> Optional[pd.DataFrame]:
        if code not in self.cache: return None
        last_end, last_df = self.cache[code]
        if last_df is None or last_end is None: return None
        # 窗口检查：若 end_dt 未早于已缓存的 end_dt，直接复用最后 win 行
        if end_dt >= last_end and len(last_df) >= win:
            return last_df.tail(win)
        return None
    def put(self, code: str, end_dt: pd.Timestamp, df: pd.DataFrame):
        self.cache[code] = (end_dt, df.copy())