# coding: utf-8
"""
高速版: 生成 features_daily.h5 + scaler.pkl
提速要点:
1) 对每只股票一次性计算全量特征 -> 缓存
2) 采样日仅做索引切片 + 拼接 + 一次性 mad_clip
3) 线程并行计算(可切换进程)
"""
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
# 如需更强并行, 可改为: from concurrent.futures import ProcessPoolExecutor as PoolExec
from config import CFG
from utils import mad_clip, Scaler, weekly_fridays, load_calendar
from utils import pkl_dump

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

def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    ret = close.diff()
    up = ret.clip(lower=0.0); dn = (-ret).clip(lower=0.0)
    ema_up = _ema(up, n); ema_dn = _ema(dn, n)
    rs = ema_up / (ema_dn + EPS)
    return 100 - 100 / (1 + rs)

def _macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = _ema(close, fast); ema_slow = _ema(close, slow)
    dif = ema_fast - ema_slow; dea = _ema(dif, signal); hist = dif - dea
    return dif, dea, hist

def _boll(close: pd.Series, n=20, k=2.0):
    ma = _mean_rolling(close, n); sd = _std_rolling(close, n)
    upper = ma + k * sd; lower = ma - k * sd
    width = (upper - lower) / (ma + EPS)
    pctb = (close - lower) / (upper - lower + EPS)
    return ma, upper, lower, width, pctb

def _kurtosis(series: pd.Series, win: int) -> pd.Series: return series.rolling(win, min_periods=1).kurt()
def _skew(series: pd.Series, win: int) -> pd.Series: return series.rolling(win, min_periods=1).skew()
def _corr(a: pd.Series, b: pd.Series, win: int) -> pd.Series: return a.rolling(win, min_periods=2).corr(b)
def _zscore(s: pd.Series, win: int) -> pd.Series:
    m = _mean_rolling(s, win); sd = _std_rolling(s, win)
    return (s - m) / (sd + EPS)
def _shifted_corr(a: pd.Series, b: pd.Series, win: int, lag: int) -> pd.Series:
    return _corr(a, b.shift(-lag), win)

def calc_factors_one_stock_full(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入: 单只股票全量日频数据(index=date)
    输出: 与df.index对齐的特征DataFrame（全量，不截断）
    """
    out = pd.DataFrame(index=df.index)
    O, H, L, C = df['open'], df['high'], df['low'], df['close']
    V, NT, TT = df['volume'], df['num_trades'], df['total_turnover']

    ret = _returns(C)
    out['ret'] = ret
    out['ret_abs'] = ret.abs()

    ema5_c = _ema(C, 5)
    ma20_v = _mean_rolling(V.replace(0, np.nan), 20)
    out['pvo'] = (C - ema5_c) / (ma20_v + EPS)

    max20_c = _max_rolling(C, 20); min20_c = _min_rolling(C, 20)
    out['sse'] = (max20_c - min20_c) / (ma20_v + EPS)

    std20_v = _std_rolling(V, 20)
    out['liq'] = (V - V.shift(1)).abs() / (std20_v + EPS)

    out['skew'] = _skew(ret, 20)
    out['gap']  = (O - C.shift(1)) / (C.shift(1) + EPS)

    std10_ret = _std_rolling(ret, 10); std30_ret = _std_rolling(ret, 30)
    out['vts'] = std10_ret / (std30_ret + EPS)

    out['vcll'] = _safe_div(V, ma20_v)

    std20_ret = _std_rolling(ret, 20)
    out['rvr'] = ret / (std20_ret + EPS)

    ma60_c = _mean_rolling(C, 60)
    out['csl'] = (C - ma60_c) / (ma60_c + EPS)

    out['rev'] = _corr(ret, ret.shift(1), 20)

    std5_ret = _std_rolling(ret, 5)
    out['vcbl'] = std5_ret / (std20_ret + EPS)

    out['kurtosis'] = _kurtosis(ret, 20)

    liq = out['liq']
    out['lip'] = 0.5 * liq + 0.3 * liq.shift(1) + 0.2 * liq.shift(2)

    out['vol_per_trade'] = V / (NT.replace(0, np.nan) + EPS)

    for n in [5, 10, 20, 30, 60]:
        out[f'ma{n}'] = _mean_rolling(C, n)
        out[f'ema{n}'] = _ema(C, n)
        out[f'price_ma{n}_ratio'] = C / (out[f'ma{n}'] + EPS)
        out[f'ma{n}_slope'] = out[f'ma{n}'].pct_change()

    out['ma5_10_cross']  = (out['ma5']  - out['ma10']) / (out['ma10'] + EPS)
    out['ma10_20_cross'] = (out['ma10'] - out['ma20']) / (out['ma20'] + EPS)
    out['ma20_60_cross'] = (out['ma20'] - out['ma60']) / (out['ma60'] + EPS)

    out['rsi14'] = _rsi(C, 14)
    out['rsi5']  = _rsi(C, 5)

    dif, dea, hist = _macd(C, 12, 26, 9)
    out['macd_dif']  = dif
    out['macd_dea']  = dea
    out['macd_hist'] = hist

    boll_ma, boll_up, boll_dn, boll_w, boll_pctb = _boll(C, 20, 2.0)
    out['boll_ma20']  = boll_ma
    out['boll_width'] = boll_w
    out['boll_pctb']  = boll_pctb

    tr = _true_range(H, L, C)
    out['atr14'] = _mean_rolling(tr, 14)
    out['hl_range'] = (H - L) / (C.shift(1).abs() + EPS)

    body = (C - O).abs()
    upper_shadow = (H - C.where(C >= O, O)).clip(lower=0)
    lower_shadow = (O.where(C >= O, C) - L).clip(lower=0)
    out['candle_body']  = body
    out['candle_upper'] = upper_shadow
    out['candle_lower'] = lower_shadow
    out['body_ratio']   = body / (H - L + EPS)
    out['upper_ratio']  = upper_shadow / (H - L + EPS)
    out['lower_ratio']  = lower_shadow / (H - L + EPS)

    out['turnover'] = TT
    out['turnover_ma20_ratio'] = TT / (_mean_rolling(TT, 20) + EPS)
    out['turnover_std20_ratio'] = _std_rolling(TT, 20) / (TT.abs() + EPS)
    out['price_turnover_corr20'] = _corr(C.pct_change(), TT.pct_change(), 20)
    out['volume_price_corr20']   = _corr(V.pct_change(), C.pct_change(), 20)

    out['vol_std20_over_mean20'] = _std_rolling(V, 20) / (_mean_rolling(V, 20) + EPS)
    out['vol_cv_20'] = out['vol_std20_over_mean20']

    for n in [5, 10, 20, 30, 60]:
        r = C.pct_change(n)
        out[f'mom{n}']   = r
        out[f'mom{n}_z'] = _zscore(r, 60)

    for n in [5, 10, 20, 30, 60]:
        out[f'ret_std{n}']  = _std_rolling(ret, n)
        out[f'ret_mean{n}'] = _mean_rolling(ret, n)

    for n in [5, 10, 20, 30, 60]:
        out[f'turnover_mean_ratio_{n}'] = _mean_rolling(TT, n) / (TT + EPS)
        out[f'turnover_std_ratio_{n}']  = _std_rolling(TT, n) / (TT.abs() + EPS)

    for n in [20, 60, 120]:
        roll_max = _max_rolling(C, n); roll_min = _min_rolling(C, n)
        out[f'price_pos_{n}'] = (C - roll_min) / (roll_max - roll_min + EPS)

    out['vol_ret_corr20_lead1'] = _shifted_corr(V.pct_change(), ret, 20, lag=1)
    out['vol_ret_corr20_lag1']  = _corr(V.pct_change(), ret.shift(1), 20)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out

def _compute_one(stock: str, df_all: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
    """
    给线程/进程池用：对单只股票计算全量因子
    """
    try:
        hist = df_all.loc[stock]
        # 保证为 DataFrame
        if isinstance(hist, pd.Series):
            hist = hist.to_frame().T
        hist = hist.sort_index()
        fct = calc_factors_one_stock_full(hist)
        return stock, fct
    except Exception:
        return stock, pd.DataFrame()

def fit_scaler_h5(h5_path: Path) -> Scaler:
    scaler = Scaler()
    n_seen = 0
    mean_acc = None
    m2_acc = None

    with h5py.File(h5_path, "r") as h5f:
        keys = list(h5f.keys())
        for k in tqdm(keys, desc="拟合 Scaler (增量)"):
            if "factor" not in h5f[k]:
                continue
            arr = h5f[k]["factor"][:]       # [N,T,C]
            arr = arr.reshape(-1, arr.shape[-1])   # 合并 batch & time → [N*T, C]
            if mean_acc is None:
                mean_acc = np.nansum(arr, 0)
                m2_acc   = np.nansum(arr**2, 0)
            else:
                mean_acc += np.nansum(arr, 0)
                m2_acc   += np.nansum(arr**2, 0)
            n_seen += arr.shape[0]

    scaler.mean = (mean_acc / n_seen).reshape(1, 1, -1)
    var = (m2_acc / n_seen) - np.square(scaler.mean.squeeze())
    scaler.std = (np.sqrt(var) + 1e-6).reshape(1, 1, -1)
    return scaler

def main():
    # 1) 读取日频数据
    day_df = pd.read_parquet(CFG.price_day_file)  # MultiIndex (order_book_id, date)
    day_df = day_df.sort_index()
    stocks = day_df.index.get_level_values(0).unique().tolist()

    # 2) 周五采样
    calendar = load_calendar(CFG.trading_day_file)
    fridays  = weekly_fridays(calendar)
    fridays  = fridays[(fridays >= pd.Timestamp(CFG.start_date)) & (fridays <= pd.Timestamp(CFG.end_date))]

    # 3) 并行计算所有股票的“全量因子”
    stock_feat: Dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=CFG.num_workers) as ex:
        futures = {ex.submit(_compute_one, s, day_df): s for s in stocks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="并行计算全量因子"):
            s = futures[fut]
            stk, fct = fut.result()
            if not fct.empty:
                stock_feat[stk] = fct

    if len(stock_feat) == 0:
        raise RuntimeError("未能计算出任何股票的全量因子，请检查原始数据。")

    # 4) 确定列名（与之前一致）
    sample_stock = next(iter(stock_feat.keys()))
    factor_cols  = stock_feat[sample_stock].columns.tolist()

    # 5) 写 HDF5（按采样周五分组），构建 [N, T, C]
    str_dt = h5py.string_dtype(encoding='utf-8')
    with h5py.File(CFG.feat_file, "w") as h5f:
        h5f.attrs['factor_cols'] = np.asarray(factor_cols, dtype=str_dt)

        for date_idx, d in enumerate(tqdm(fridays, desc="写入特征仓")):
            feats_all = []
            stk_list  = []
            for stk, fct_full in stock_feat.items():
                fct_hist = fct_full[fct_full.index <= d]
                if len(fct_hist) < CFG.daily_window:
                    continue
                fct_win = fct_hist.tail(CFG.daily_window)
                fct_win = fct_win.reindex(columns=factor_cols)
                vals = fct_win.values
                if not np.isfinite(vals).any() or np.isnan(vals).all():
                    continue
                vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
                if vals.ndim != 2 or vals.shape[0] != CFG.daily_window:
                    continue
                feats_all.append(vals.astype(np.float32))
                stk_list.append(stk)

            if not feats_all:
                continue

            feats_arr = np.stack(feats_all, axis=0)  # [N,T,C]
            feats_arr = mad_clip(feats_arr)
            feats_arr = np.nan_to_num(feats_arr, nan=0.0, posinf=0.0, neginf=0.0)

            g = h5f.create_group(f"date_{date_idx}")
            g.attrs['date'] = d.strftime('%Y-%m-%d')
            g.create_dataset("stocks", data=np.asarray(stk_list, dtype=str_dt))

            # 关键：为按行（样本）读取优化 chunk，并使用更快压缩
            N, T, C = feats_arr.shape
            chunk0 = int(min(256, N))  # 每块最多 256 行
            try:
                g.create_dataset(
                    "factor",
                    data=feats_arr,
                    compression="lzf",        # 更快（若不可用，可用 None）
                    chunks=(chunk0, T, C)
                )
            except Exception:
                g.create_dataset(
                    "factor",
                    data=feats_arr,
                    compression=None,         # 退化为不压缩，读取最快
                    chunks=(chunk0, T, C)
                )

    # 6) 拟合 scaler（所有日期合并）
    # 调用
    scaler = fit_scaler_h5(CFG.feat_file)
    pkl_dump(scaler, CFG.scaler_file)
    print("Scaler 拟合完成！")

if __name__ == "__main__":
    main()
    # scaler = fit_scaler_h5(CFG.feat_file)
    # pkl_dump(scaler, CFG.scaler_file)
    # print("Scaler 拟合完成！")