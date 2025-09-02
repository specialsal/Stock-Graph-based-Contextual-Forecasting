# coding: utf-8
"""
30分钟级特征工程: 生成 features_30m.h5 + scaler_30m.pkl

设计要点:
1) 对每只股票一次性计算全量高频因子(30m级, 与 datetime 对齐) -> 缓存
2) 以“交易日收盘最后一个30m bar”为采样点：仅做索引切片 + 拼接 + 一次性 mad_clip
3) 线程并行计算(可切换进程池)
4) 因子不超过20个，突出高频(日内/跨日)特征

输入:
- stock_price_30m.parquet: MultiIndex(order_book_id, datetime)
  列: ['open','high','low','close','volume','num_trades','total_turnover']

输出:
- features_30m.h5: 按交易日分组存储 [N, T, C] 特征张量
- scaler_30m.pkl : 针对合并样本拟合的标准化器(依赖 utils.Scaler)

注意:
- 如无 utils.load_calendar，可从 data/trading_calendar.csv 等载入；本文件内提供从 30m 数据直接提取交易日日历的方式。
"""

import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed

# 如果你更习惯用进程池，请改:
# from concurrent.futures import ProcessPoolExecutor as ThreadPoolExecutor

# ======================
# 本文件内的简易配置
# ======================
class CFG_30M:
    # 文件路径
    price_30m_file = Path("data/raw/stock_price_30m.parquet")
    feat_file      = Path("data/processed/features_30m.h5")
    scaler_file    = Path("data/processed/scaler_30m.pkl")

    # 时间范围（字符串或 None）
    start_date = "2011-01-01"  # 如 "2012-01-01"
    end_date   = "2025-08-26"  # 如 "2025-08-26"

    # 并行
    num_workers = 8

    # 窗口长度(以30m bar计): 例如 160 ≈ 20个交易日 * 8bar/日
    daily_window_30m = 160

    # 是否只取日内交易时段(若有夜盘则可再定制)
    intraday_only = True

    # 交易日过滤: 例如剔除成交基本为0的日子
    min_bars_per_day = 4  # 少于该bar数的交易日可忽略(视数据而定)

# ==============
# 依赖工具函数
# ==============
EPS = 1e-12

def _safe_div(a, b): return a / (b.replace(0, np.nan) + EPS)
def _ema(s: pd.Series, span: int) -> pd.Series: return s.ewm(span=span, adjust=False, min_periods=1).mean()
def _std_rolling(s: pd.Series, win: int) -> pd.Series: return s.rolling(win, min_periods=1).std()
def _mean_rolling(s: pd.Series, win: int) -> pd.Series: return s.rolling(win, min_periods=1).mean()
def _max_rolling(s: pd.Series, win: int) -> pd.Series: return s.rolling(win, min_periods=1).max()
def _min_rolling(s: pd.Series, win: int) -> pd.Series: return s.rolling(win, min_periods=1).min()
def _returns(close: pd.Series) -> pd.Series: return close.pct_change()
def _true_range(high, low, close):
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr
def _corr(a: pd.Series, b: pd.Series, win: int) -> pd.Series: return a.rolling(win, min_periods=2).corr(b)

def trading_day_from_intraday(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.to_datetime(idx.date)

def calc_factors_one_stock_30m_full(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    O, H, L, C = df['open'], df['high'], df['low'], df['close']
    V, NT, TT = df['volume'], df['num_trades'], df['total_turnover']

    ret_1 = C.pct_change()
    ret_4 = C.pct_change(4)
    out['ret_30m'] = ret_1
    out['ret_2h']  = ret_4

    out['vol_8']  = _std_rolling(ret_1, 8)
    out['vol_40'] = _std_rolling(ret_1, 40)
    out['vol_ratio_8_40'] = _safe_div(out['vol_8'], out['vol_40'])

    tr = _true_range(H, L, C)
    out['atr_8']  = _mean_rolling(tr, 8)
    out['hl_rng'] = _safe_div(H - L, C.shift(1).abs())

    out['mom_3'] = C.pct_change(3)
    out['mom_8'] = C.pct_change(8)

    body = (C - O).abs()
    up_shadow = (H - C.where(C >= O, O)).clip(lower=0)
    lo_shadow = (O.where(C >= O, C) - L).clip(lower=0)
    rng = (H - L).replace(0, np.nan)
    out['body_ratio']  = body / (rng + EPS)
    out['upper_ratio'] = up_shadow / (rng + EPS)
    out['lower_ratio'] = lo_shadow / (rng + EPS)

    v_ma20 = _mean_rolling(V.replace(0, np.nan), 20)
    out['v_ma_ratio'] = _safe_div(V, v_ma20)
    out['v_cv_20']    = _safe_div(_std_rolling(V, 20), _mean_rolling(V, 20))
    out['nt_per_bar'] = (df['num_trades']).fillna(0.0)
    out['v_per_trade'] = _safe_div(V, df['num_trades'].replace(0, np.nan))

    out['corr_v_ret_20'] = _corr(V.pct_change(), ret_1, 20)

    ema12 = _ema(C, 12)
    ema48 = _ema(C, 48)
    out['ema12_dev'] = _safe_div(C - ema12, ema12.abs())
    out['ema48_dev'] = _safe_div(C - ema48, ema48.abs())

    m20 = _mean_rolling(C, 20)
    sd20 = _std_rolling(C, 20)
    out['boll_w_20'] = _safe_div(2 * sd20, m20.abs())

    out['overnight_gap'] = np.nan

    out = out.replace([np.inf, -np.inf], np.nan)
    return out

def _compute_one(stock: str, df_all: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
    try:
        hist = df_all.loc[stock]
        if isinstance(hist, pd.Series):
            hist = hist.to_frame().T
        hist = hist.sort_index()
        fct = calc_factors_one_stock_30m_full(hist)
        return stock, fct
    except Exception:
        return stock, pd.DataFrame()

try:
    from utils import mad_clip, Scaler, pkl_dump
except Exception:
    def mad_clip(x: np.ndarray, c=5.0) -> np.ndarray:
        med = np.nanmedian(x, axis=0, keepdims=True)
        mad = np.nanmedian(np.abs(x - med), axis=0, keepdims=True) + EPS
        z = (x - med) / (1.4826 * mad)
        z = np.clip(z, -c, c)
        return z
    class Scaler:
        def __init__(self): self.mean_=None; self.std_=None
        def fit(self, arr: np.ndarray):
            self.mean_ = np.nanmean(arr, axis=(0,1), keepdims=True)
            self.std_  = np.nanstd(arr, axis=(0,1), keepdims=True) + EPS
        def transform(self, arr: np.ndarray) -> np.ndarray:
            return (arr - self.mean_) / self.std_
        def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
            return arr * self.std_ + self.mean_
    import pickle
    def pkl_dump(obj, path: Path):
        with open(path, "wb") as f:
            pickle.dump(obj, f)

def normalize_to_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    """
    将输入 df 规范为 MultiIndex(order_book_id, datetime) 且 datetime 为升序。
    允许以下情况：
    - 已是 MultiIndex 并包含上述两个层
    - 单层索引 + 列中另一个键
    - 两个键都在列里
    """
    df = df.copy()

    # 情况1：已是 MultiIndex
    if isinstance(df.index, pd.MultiIndex):
        names = list(df.index.names)
        # 如果层名字不一致，按位置假定 level0=order_book_id, level1=datetime
        if len(df.index.levels) != 2:
            raise ValueError("期望二维 MultiIndex，但检测到层数 != 2")
        # 尝试重命名
        if set(names) != set(['order_book_id', 'datetime']):
            # 不管当前叫什么，重命名为标准名
            df.index = df.index.set_names(['order_book_id', 'datetime'])
        # 确保第二层是 DatetimeIndex
        lvl1 = df.index.get_level_values(1)
        if not np.issubdtype(lvl1.dtype, np.datetime64):
            df = df.reset_index()
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index(['order_book_id', 'datetime']).sort_index()
        else:
            df = df.sort_index()
        return df

    # 情况2：单层索引 + 列中包含另一个键
    idx_name = df.index.name
    cols = df.columns

    if idx_name == 'order_book_id' and 'datetime' in cols:
        df = df.reset_index()
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index(['order_book_id', 'datetime']).sort_index()
        return df

    if (idx_name == 'datetime' or np.issubdtype(df.index.dtype, np.datetime64)) and 'order_book_id' in cols:
        df = df.reset_index()
        df['datetime'] = pd.to_datetime(df['datetime'])  # 确保时间类型
        df = df.set_index(['order_book_id', 'datetime']).sort_index()
        return df

    # 情况3：两个键都在列里
    if 'order_book_id' in cols and 'datetime' in cols:
        df = df.reset_index(drop=True)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index(['order_book_id', 'datetime']).sort_index()
        return df

    # 无法规范化
    raise ValueError("无法识别索引，请确保存在 order_book_id 与 datetime 字段（在索引或列中均可）。")
def compute_factors_batch(df: pd.DataFrame) -> pd.DataFrame:
    # df: MultiIndex(order_book_id, datetime)，带列：open,high,low,close,volume,num_trades,total_turnover
    # 返回：与 df 对齐的同索引 DataFrame（因子列）
    out = pd.DataFrame(index=df.index)

    def g_transform(s, func, win=None, span=None):
        if win is not None:
            return s.groupby(level=0).apply(lambda x: x.rolling(win, min_periods=1).apply(func, raw=False))
        if span is not None:
            return s.groupby(level=0).apply(lambda x: x.ewm(span=span, adjust=False, min_periods=1).mean())
        raise ValueError

    # 简洁起见，这里用常见写法（注意：为了性能更好，尽量用 transform + 内置 rolling/std/mean）
    # 先分组对象
    grp = df.groupby(level=0, sort=False)

    C = df['close']; O = df['open']; H = df['high']; L = df['low']
    V = df['volume']; NT = df['num_trades']; TT = df['total_turnover']

    # returns
    ret_1 = grp['close'].pct_change()
    ret_4 = grp['close'].pct_change(4)
    out['ret_30m'] = ret_1
    out['ret_2h']  = ret_4

    # vol windows
    vol_8  = ret_1.groupby(level=0).rolling(8, min_periods=1).std().reset_index(level=0, drop=True)
    vol_40 = ret_1.groupby(level=0).rolling(40, min_periods=1).std().reset_index(level=0, drop=True)
    out['vol_8']  = vol_8
    out['vol_40'] = vol_40
    out['vol_ratio_8_40'] = vol_8 / (vol_40.replace(0, np.nan) + EPS)

    # TR / ATR
    prev_close = grp['close'].shift(1)
    tr = pd.concat([H - L, (H - prev_close).abs(), (L - prev_close).abs()], axis=1).max(axis=1)
    atr_8 = tr.groupby(level=0).rolling(8, min_periods=1).mean().reset_index(level=0, drop=True)
    out['atr_8']  = atr_8
    out['hl_rng'] = (H - L) / (C.shift(1).groupby(level=0).transform('first').abs() + EPS)  # 简化，或直接用 prev_close.abs()

    # momentum
    out['mom_3'] = grp['close'].pct_change(3)
    out['mom_8'] = grp['close'].pct_change(8)

    # candle body/shadow
    body = (C - O).abs()
    up_shadow = (H - C.where(C >= O, O)).clip(lower=0)
    lo_shadow = (O.where(C >= O, C) - L).clip(lower=0)
    rng = (H - L).replace(0, np.nan)
    out['body_ratio']  = body / (rng + EPS)
    out['upper_ratio'] = up_shadow / (rng + EPS)
    out['lower_ratio'] = lo_shadow / (rng + EPS)

    # volume features
    v_ma20 = V.replace(0, np.nan).groupby(level=0).rolling(20, min_periods=1).mean().reset_index(level=0, drop=True)
    v_std20 = V.groupby(level=0).rolling(20, min_periods=1).std().reset_index(level=0, drop=True)
    v_mean20 = V.groupby(level=0).rolling(20, min_periods=1).mean().reset_index(level=0, drop=True)
    out['v_ma_ratio'] = V / (v_ma20 + EPS)
    out['v_cv_20']    = v_std20 / (v_mean20 + EPS)
    out['nt_per_bar'] = NT.fillna(0.0)
    out['v_per_trade'] = V / (NT.replace(0, np.nan) + EPS)

    # corr(V, ret, 20)
    vret = V.groupby(level=0).pct_change()
    corr_20 = vret.groupby(level=0).rolling(20, min_periods=2).corr(ret_1).reset_index(level=0, drop=True)
    out['corr_v_ret_20'] = corr_20

    # EMA dev
    ema12 = grp['close'].apply(lambda x: x.ewm(span=12, adjust=False, min_periods=1).mean())
    ema48 = grp['close'].apply(lambda x: x.ewm(span=48, adjust=False, min_periods=1).mean())
    out['ema12_dev'] = (C - ema12) / (ema12.abs() + EPS)
    out['ema48_dev'] = (C - ema48) / (ema48.abs() + EPS)

    # Boll width
    m20 = grp['close'].rolling(20, min_periods=1).mean().reset_index(level=0, drop=True)
    sd20 = grp['close'].rolling(20, min_periods=1).std().reset_index(level=0, drop=True)
    out['boll_w_20'] = (2 * sd20) / (m20.abs() + EPS)

    # overnight gap: 预计算为每行可用的列
    # 当行在“当日最后bar”时，gap = 次日首开/当日末收 - 1
    # 提前构造 next_day_first_open 与 day_last_close
    # 1) 找到当日最后收盘价
    day_last_close = df['close'].where(df['is_day_last']).groupby(['order_book_id', 'trade_day']).transform('max')
    # 实际上上面 transform('max') 等价于把最后bar的 close 向整日广播
    # 2) 次日首开，先把每日首开取出来，再对 trade_day 做 +1 日的对齐
    day_first_open = df['open'].where(df['is_day_first']).groupby(['order_book_id', 'trade_day']).transform('max')

    # 索引到“本行 trade_day 的次日首开”
    # 先得到一个 per-row 的 next_day 键
    next_day = df['trade_day'] + pd.Timedelta(days=1)
    key_curr = pd.MultiIndex.from_arrays([df.index.get_level_values('order_book_id'), df['trade_day']])
    key_next = pd.MultiIndex.from_arrays([df.index.get_level_values('order_book_id'), next_day])

    # 构造映射: (stk, day) -> 当日首开/末收
    # 用 groupby.first/last 更稳，当天多个 bar 时只取首/末
    day_open = df.loc[df['is_day_first'], ['open','trade_day']].reset_index().set_index(['order_book_id','trade_day'])['open']
    day_close = df.loc[df['is_day_last'],  ['close','trade_day']].reset_index().set_index(['order_book_id','trade_day'])['close']

    curr_day_last_close = day_close.reindex(key_curr)
    next_day_first_open = day_open.reindex(key_next)

    overnight_gap = (next_day_first_open.values - curr_day_last_close.values) / (np.abs(curr_day_last_close.values) + EPS)
    # 只有当日最后bar那一行设 gap，其它行为 NaN（避免污染窗口）
    out['overnight_gap'] = np.nan
    mask_last = df['is_day_last'].values
    out.loc[mask_last, 'overnight_gap'] = overnight_gap[mask_last]

    # 清理
    out = out.replace([np.inf, -np.inf], np.nan)
    return out
def main():
    cfg = CFG_30M

    print("读取 30m 数据 ...")
    df = pd.read_parquet(cfg.price_30m_file)
    # 规范为 MultiIndex(order_book_id, datetime)
    df = normalize_to_multiindex(df)

    df = df.sort_index()

    # 添加 trade_day
    print("添加交易日字段 ...")
    dt = df.index.get_level_values('datetime')
    df['trade_day'] = pd.to_datetime(dt.date)
    
    # 日内排序编号（每只股票、每天从 0 开始）
    print("添加日内排序编号 ...")
    df['bar_in_day'] = (
        df.groupby(['order_book_id', 'trade_day']).cumcount()
    )
    # 每日最后一个bar的编号
    print("标记日内首尾bar ...")
    last_bar_in_day = df.groupby(['order_book_id', 'trade_day'])['bar_in_day'].transform('max')
    df['is_day_last']  = (df['bar_in_day'] == last_bar_in_day)
    df['is_day_first'] = (df['bar_in_day'] == 0)

    needed_cols = ['open','high','low','close','volume','num_trades','total_turnover']
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要字段: {missing}")

    # 时间范围过滤（使用同一 DataFrame 的索引进行筛选，避免长度不一致）
    print("时间范围过滤 ...")
    if cfg.start_date is not None or cfg.end_date is not None:
        # 用 .loc 在 MultiIndex 上按第二层 datetime 范围切片，要求整体按索引排序
        df = df.sort_index()
        start = pd.Timestamp(cfg.start_date) if cfg.start_date is not None else None
        end   = pd.Timestamp(cfg.end_date)   if cfg.end_date   is not None else None

        # 如果 start/end 都有，优先用 slice；否则分别用条件筛选
        if start is not None and end is not None:
            df = df.loc[(slice(None), slice(start, end)), :]
        elif start is not None:
            df = df.loc[(slice(None), slice(start, None)), :]
        elif end is not None:
            df = df.loc[(slice(None), slice(None, end)), :]

    # 交易日映射
    print("提取交易日日历 ...")
    dt = df.index.get_level_values('datetime')
    trade_day = trading_day_from_intraday(dt)
    df = df.copy()
    df['trade_day'] = trade_day.normalize()

    print("批量计算因子 ...")
    factors_df = compute_factors_batch(df[[
        'open','high','low','close','volume','num_trades','total_turnover',
        'trade_day','bar_in_day','is_day_last','is_day_first'
    ]].drop(columns=['trade_day','bar_in_day','is_day_last','is_day_first'], errors='ignore'))

    factor_cols = factors_df.columns.tolist()
    str_dt = h5py.string_dtype(encoding='utf-8')

    # 为每只股票创建“位置索引”以便 O(1) 回看窗口
    print("构建位置索引 ...")
    df_reset = df.reset_index()  # columns: order_book_id, datetime, ...
    # 每只股票的行在 MultiIndex 中是连续的（已 sort_index）
    # 建一个 per-stock 的 numpy 索引映射
    stock_groups = df_reset.groupby('order_book_id', sort=False)
    stock_pos = {}  # stk -> array of datetimes (aligned)
    for stk, sub in stock_groups:
        stock_pos[stk] = sub['datetime'].to_numpy()

    # 采样点：所有日末 bar 的行索引
    idx_sample = df.index[df['is_day_last']]
    samples_meta = df.loc[idx_sample, ['trade_day']].reset_index()

    with h5py.File(CFG_30M.feat_file, "w") as h5f:
        h5f.attrs['factor_cols'] = np.asarray(factor_cols, dtype=str_dt)

        # 遍历每个交易日的样本（向量化按日分组）
        for date_idx, (d, meta_day) in enumerate(tqdm(samples_meta.groupby('trade_day'), desc="写入特征")):
            # meta_day: 行包含该日所有股票的最后bar行的 (order_book_id, datetime)
            feats_all = []
            stk_list = []

            for _, row in meta_day.iterrows():
                stk = row['order_book_id']
                ts_last = row['datetime']

                # 找到该 ts_last 在该股时间轴中的位置
                times = stock_pos[stk]
                pos = np.searchsorted(times, ts_last)
                # 取窗口 [pos - T, pos)
                T = CFG_30M.daily_window_30m
                if pos < T:
                    continue
                idx_slice = pd.MultiIndex.from_product([[stk], times[pos - T:pos]])
                fct_win = factors_df.loc[idx_slice, factor_cols]

                vals = fct_win.values
                if not np.isfinite(vals).any() or np.isnan(vals).all():
                    continue
                feats_all.append(np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))
                stk_list.append(stk)

            if not feats_all:
                continue

            feats_arr = np.stack(feats_all, axis=0)
            feats_arr = mad_clip(feats_arr)
            feats_arr = np.nan_to_num(feats_arr, nan=0.0, posinf=0.0, neginf=0.0)

            g = h5f.create_group(f"date_{date_idx}")
            g.attrs['date'] = pd.Timestamp(d).strftime('%Y-%m-%d')
            g.create_dataset("stocks", data=np.asarray(stk_list, dtype=str_dt))
            g.create_dataset("factor", data=feats_arr, compression="gzip")

    print("拟合 scaler_30m.pkl ...")
    samples = []
    with h5py.File(cfg.feat_file, "r") as h5f:
        for k in h5f.keys():
            if 'factor' in h5f[k]:
                samples.append(h5f[k]['factor'][:])
    if len(samples) == 0:
        raise RuntimeError("未生成任何特征数据，请检查时间范围与股票数。")

    all_arr = np.concatenate(samples, axis=0)
    scaler = Scaler()
    scaler.fit(all_arr)
    pkl_dump(scaler, cfg.scaler_file)

    print("30m 特征与 scaler 生成完成！")
    print(f"特征维度 C = {all_arr.shape[-1]}, 每只股票序列长度 T = {all_arr.shape[1]}")
    print(f"覆盖股票数: {len(df.index.get_level_values('order_book_id').unique())}, 采样交易日数量: {len(samples)}")

if __name__ == "__main__":
    main()