# coding: utf-8
"""
生成 5min 聚合的日频序列特征：features_5m.h5 + scaler_5m.pkl

优化与修复
- I/O：pyarrow.dataset 按批流式读取，仅选择 close 与 total_turnover 两列（可根据需要扩展）。
- 计算：NumPy + numba（float32），按日聚合。对异常/脏数据更强防御：
  * total_turnover <= TO_EPS 视为无效，参与 ILLIQ 的分母一定>0；
  * ILLIQ 内核跳过 tt<=0；
  * 仅在有效成交额数>=3时计算“成交额大段”分位数与相关特征；
  * 严禁对全 NaN/无效数组做 quantile，零警告。
- 并行：以“批”为任务，进程池 + wait(FIRST_COMPLETED)，无 TimeoutError。
- 精度：全流程 float32，HDF5 保存 float32，降低内存/I/O。
"""

from typing import Dict, List
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute as pc

from concurrent.futures import ProcessPoolExecutor as PoolExec
from concurrent.futures import wait, FIRST_COMPLETED

from config import CFG
from utils import mad_clip, Scaler, load_calendar, weekly_fridays, pkl_dump

# ============ 可调参数 ============
NEEDED_COLS = ['close', 'total_turnover']  # 最小必要集合，最快
# 若确需其它列，请改为：['open','high','low','close','volume','total_turnover','num_trades']
BATCH_STOCK_SIZE_DEFAULT = 512
MAX_INFLIGHT_DEFAULT = 4
TO_EPS = np.float32(1e-12)   # 成交额有效阈值；<= TO_EPS 视为无效
# =================================

# -----------------------------
# 数值核（numba / float32）
# -----------------------------
from numba import njit

EPS32 = np.float32(1e-12)

@njit(cache=True, fastmath=True)
def illiq_np32(ret: np.ndarray, to: np.ndarray) -> np.float32:
    # 防御：跳过非有限或分母<=0的数据点
    n = ret.shape[0]
    s = np.float32(0.0)
    cnt = 0
    for i in range(n):
        rr = ret[i]
        tt = to[i]
        if np.isfinite(rr) and np.isfinite(tt) and tt > EPS32:
            s += np.abs(rr) / (tt + EPS32)
            cnt += 1
    return s / cnt if cnt > 0 else np.float32(np.nan)

@njit(cache=True, fastmath=True)
def mdd_np32(ret: np.ndarray) -> np.float32:
    n = ret.shape[0]
    peak = np.float32(1.0)
    equity = np.float32(1.0)
    min_dd = np.float32(0.0)
    for i in range(n):
        r = ret[i] if np.isfinite(ret[i]) else np.float32(0.0)
        equity *= (np.float32(1.0) + r)
        if equity > peak:
            peak = equity
        dd = equity / (peak + EPS32) - np.float32(1.0)
        if dd < min_dd:
            min_dd = dd
    return min_dd if n > 0 else np.float32(np.nan)

@njit(cache=True, fastmath=True)
def corr_np32(x: np.ndarray, y: np.ndarray) -> np.float32:
    n = x.shape[0]
    cnt = 0
    for i in range(n):
        if np.isfinite(x[i]) and np.isfinite(y[i]):
            cnt += 1
    if cnt < 2:
        return np.float32(np.nan)
    xs = np.empty(cnt, dtype=np.float32)
    ys = np.empty(cnt, dtype=np.float32)
    k = 0
    for i in range(n):
        if np.isfinite(x[i]) and np.isfinite(y[i]):
            xs[k] = x[i]; ys[k] = y[i]; k += 1
    mx = np.float32(0.0); my = np.float32(0.0)
    for i in range(cnt):
        mx += xs[i]; my += ys[i]
    mx /= cnt; my /= cnt
    vx = np.float32(0.0); vy = np.float32(0.0); cov = np.float32(0.0)
    for i in range(cnt):
        dx = xs[i] - mx; dy = ys[i] - my
        cov += dx * dy; vx += dx * dx; vy += dy * dy
    if vx <= 0 or vy <= 0:
        return np.float32(np.nan)
    return cov / np.sqrt(vx * vy)

@njit(cache=True, fastmath=True)
def kurtosis_np32(x: np.ndarray) -> np.float32:
    n = 0
    for i in range(x.shape[0]):
        if np.isfinite(x[i]):
            n += 1
    if n < 4:
        return np.float32(np.nan)
    vals = np.empty(n, dtype=np.float32)
    k = 0
    for i in range(x.shape[0]):
        if np.isfinite(x[i]):
            vals[k] = x[i]; k += 1
    m = np.float32(0.0)
    for i in range(n): m += vals[i]
    m /= n
    v = np.float32(0.0)
    for i in range(n):
        d = vals[i] - m; v += d*d
    v /= (n - 1)
    if v <= 0: return np.float32(np.nan)
    m4 = np.float32(0.0)
    for i in range(n):
        d = vals[i] - m; m4 += d*d*d*d
    m4 /= n
    return m4/(v*v) - np.float32(3.0)

@njit(cache=True, fastmath=True)
def skewness_np32(x: np.ndarray) -> np.float32:
    n = 0
    for i in range(x.shape[0]):
        if np.isfinite(x[i]):
            n += 1
    if n < 3:
        return np.float32(np.nan)
    vals = np.empty(n, dtype=np.float32)
    k = 0
    for i in range(x.shape[0]):
        if np.isfinite(x[i]):
            vals[k] = x[i]; k += 1
    m = np.float32(0.0)
    for i in range(n): m += vals[i]
    m /= n
    s2 = np.float32(0.0)
    for i in range(n):
        d = vals[i] - m; s2 += d*d
    s2 /= (n - 1)
    if s2 <= 0: return np.float32(np.nan)
    s = np.sqrt(s2)
    num = np.float32(0.0)
    for i in range(n): num += ((vals[i]-m)/s)**3
    return num / n

@njit(cache=True, fastmath=True)
def path_mom_np32(ret: np.ndarray) -> np.float32:
    ssum = np.float32(0.0); asum = np.float32(0.0)
    for i in range(ret.shape[0]):
        r = ret[i] if np.isfinite(ret[i]) else np.float32(0.0)
        ssum += r; asum += abs(r)
    return ssum / (asum + EPS32)

@njit(cache=True, fastmath=True)
def std_np32(x: np.ndarray) -> np.float32:
    n = 0; mean = np.float32(0.0)
    for i in range(x.shape[0]):
        if np.isfinite(x[i]): mean += x[i]; n += 1
    if n < 2: return np.float32(np.nan)
    mean /= n
    var = np.float32(0.0)
    for i in range(x.shape[0]):
        if np.isfinite(x[i]):
            d = x[i] - mean; var += d*d
    var /= (n - 1)
    if var <= 0: return np.float32(np.nan)
    return np.sqrt(var)

@njit(cache=True, fastmath=True)
def downside_vol_ratio_np32(ret: np.ndarray) -> np.float32:
    std_all = std_np32(ret)
    if not np.isfinite(std_all) or std_all <= 0:
        return np.float32(np.nan)
    n_neg = 0
    for i in range(ret.shape[0]):
        if np.isfinite(ret[i]) and ret[i] < 0:
            n_neg += 1
    if n_neg < 2:
        return np.float32(np.nan)
    vals = np.empty(n_neg, dtype=np.float32)
    k = 0
    for i in range(ret.shape[0]):
        if np.isfinite(ret[i]) and ret[i] < 0:
            vals[k] = ret[i]; k += 1
    std_neg = std_np32(vals)
    if not np.isfinite(std_neg): return np.float32(np.nan)
    return std_neg / (std_all + EPS32)

# -----------------------------
# 单股：构建当日高频特征（NumPy/numba）
# -----------------------------
def _build_daily_hf_features(df_5m_one: pd.DataFrame, vol_top_q: float = 0.9) -> pd.DataFrame:
    ddf = df_5m_one.sort_index()
    if ddf.empty:
        return pd.DataFrame()

    # 必要列校验
    for c in ['close', 'total_turnover']:
        if c not in ddf.columns:
            return pd.DataFrame()

    # 转 float32
    close = ddf['close'].astype('float32').to_numpy(copy=False)
    to = ddf['total_turnover'].astype('float32').to_numpy(copy=False)

    # 将无效成交额（<=TO_EPS）统一标记为 NaN，确保 ILLIQ 分母不会为 0
    to = np.where((~np.isfinite(to)) | (to <= TO_EPS), np.float32(np.nan), to)

    # 5m 收益、成交额变化率（float32）
    close_shift = np.roll(close, 1)
    close_shift[0] = close[0]
    ret = close / close_shift - np.float32(1.0)
    ret[0] = np.float32(np.nan)
    # 清理非有限
    ret[~np.isfinite(ret)] = np.float32(np.nan)

    to_shift = np.roll(to, 1)
    to_shift[0] = to[0]
    to_pct = to / to_shift - np.float32(1.0)
    to_pct[0] = np.float32(np.nan)
    to_pct[~np.isfinite(to_pct)] = np.float32(np.nan)

    # 按自然日切段
    dates_np = ddf.index.normalize().values.astype('datetime64[D]')
    boundaries = np.nonzero(np.concatenate(([True], dates_np[1:] != dates_np[:-1])))[0]
    start_idxs = boundaries
    end_idxs = np.empty_like(start_idxs)
    end_idxs[:-1] = start_idxs[1:] - 1
    end_idxs[-1] = len(ddf) - 1

    rows: List[Dict[str, np.float32]] = []
    idxs: List[pd.Timestamp] = []

    for s, e in zip(start_idxs, end_idxs):
        n = e - s + 1
        if n < 5:
            continue

        r = ret[s:e+1]
        t = to[s:e+1]
        vchg = to_pct[s:e+1]

        # 有效成交额计数（严格 > TO_EPS）
        valid_to_count = int(np.sum(np.isfinite(t)))
        has_valid_to = valid_to_count > 0

        # 仅当有效成交额数>=3时计算分位数阈值，避免奇异阈值
        if valid_to_count >= 3:
            thr = np.nanpercentile(t.astype('float64'), vol_top_q * 100.0).astype('float32')
            # 防御：阈值也必须 > TO_EPS，否则 big_mask 为空
            if np.isfinite(thr) and thr > TO_EPS:
                big_mask = t >= thr
            else:
                big_mask = np.zeros(n, dtype=bool)
        else:
            big_mask = np.zeros(n, dtype=bool)

        cut = min(6, n // 4)
        idx_open = slice(0, cut)
        idx_close = slice(n - cut, n)

        def seg_corr(rr: np.ndarray, vv: np.ndarray):
            c0 = corr_np32(rr, vv)
            if rr.shape[0] >= 2:
                c_lead = corr_np32(rr[:-1], vv[:-1])
                c_lag  = corr_np32(rr[1:],  vv[1:])
            else:
                c_lead = np.float32(np.nan)
                c_lag  = np.float32(np.nan)
            return c0, c_lead, c_lag

        feat: Dict[str, np.float32] = {}

        # ILLIQ（仅在存在有效成交额时计算；numba 内核也会跳过分母<=0）
        feat['illiq_all']   = illiq_np32(r, t) if has_valid_to else np.float32(np.nan)
        feat['illiq_open']  = illiq_np32(r[idx_open], t[idx_open]) if has_valid_to and cut > 0 else np.float32(np.nan)
        feat['illiq_close'] = illiq_np32(r[idx_close], t[idx_close]) if has_valid_to and cut > 0 else np.float32(np.nan)
        feat['illiq_big']   = illiq_np32(r[big_mask], t[big_mask]) if has_valid_to and big_mask.any() else np.float32(np.nan)

        # 收益-成交额变化相关
        c_all = seg_corr(r, vchg)
        feat['ret_vol_corr']      = c_all[0]
        feat['ret_vol_corr_lead'] = c_all[1]
        feat['ret_vol_corr_lag']  = c_all[2]

        if cut > 0:
            c_open = seg_corr(r[idx_open], vchg[idx_open])
            c_close = seg_corr(r[idx_close], vchg[idx_close])
        else:
            c_open = (np.float32(np.nan),)*3
            c_close = (np.float32(np.nan),)*3

        feat['ret_vol_corr_open']      = c_open[0]
        feat['ret_vol_corr_open_lead'] = c_open[1]
        feat['ret_vol_corr_open_lag']  = c_open[2]

        feat['ret_vol_corr_close']      = c_close[0]
        feat['ret_vol_corr_close_lead'] = c_close[1]
        feat['ret_vol_corr_close_lag']  = c_close[2]

        if big_mask.any():
            c_big = seg_corr(r[big_mask], vchg[big_mask])
            feat['ret_vol_corr_big']      = c_big[0]
            feat['ret_vol_corr_big_lead'] = c_big[1]
            feat['ret_vol_corr_big_lag']  = c_big[2]
        else:
            feat['ret_vol_corr_big']      = np.float32(np.nan)
            feat['ret_vol_corr_big_lead'] = np.float32(np.nan)
            feat['ret_vol_corr_big_lag']  = np.float32(np.nan)

        # MDD
        feat['mdd_all']   = mdd_np32(r)
        feat['mdd_open']  = mdd_np32(r[idx_open]) if cut > 0 else np.float32(np.nan)
        feat['mdd_close'] = mdd_np32(r[idx_close]) if cut > 0 else np.float32(np.nan)
        feat['mdd_big']   = mdd_np32(r[big_mask]) if big_mask.any() else np.float32(np.nan)

        # Kurt/Skew
        feat['kurt_all']  = kurtosis_np32(r)
        feat['skew_all']  = skewness_np32(r)
        feat['kurt_open'] = kurtosis_np32(r[idx_open]) if cut > 0 else np.float32(np.nan)
        feat['skew_open'] = skewness_np32(r[idx_open]) if cut > 0 else np.float32(np.nan)
        feat['kurt_close'] = kurtosis_np32(r[idx_close]) if cut > 0 else np.float32(np.nan)
        feat['skew_close'] = skewness_np32(r[idx_close]) if cut > 0 else np.float32(np.nan)
        if big_mask.any():
            feat['kurt_big'] = kurtosis_np32(r[big_mask])
            feat['skew_big'] = skewness_np32(r[big_mask])
        else:
            feat['kurt_big'] = np.float32(np.nan)
            feat['skew_big'] = np.float32(np.nan)

        # 路径动量
        feat['path_mom_all']   = path_mom_np32(r)
        feat['path_mom_open']  = path_mom_np32(r[idx_open]) if cut > 0 else np.float32(np.nan)
        feat['path_mom_close'] = path_mom_np32(r[idx_close]) if cut > 0 else np.float32(np.nan)
        feat['path_mom_big']   = path_mom_np32(r[big_mask]) if big_mask.any() else np.float32(np.nan)

        # 极端收益均值（|z|>2）
        r64 = r.astype('float64', copy=False)
        mu = np.nanmean(r64).astype('float32')
        sd = np.nanstd(r64, ddof=1).astype('float32')
        if np.isfinite(sd) and sd > 0:
            mask_ext = (r > mu + 2*sd) | (r < mu - 2*sd)
            if mask_ext.any():
                feat['extret_mean_5m'] = np.nanmean(r64[mask_ext]).astype('float32')
            else:
                feat['extret_mean_5m'] = np.float32(np.nan)
        else:
            feat['extret_mean_5m'] = np.float32(np.nan)

        # 日内分段累计涨跌幅
        feat['ret_ex_open30'] = np.float32(np.nansum(r[cut:])) if cut > 0 else np.float32(np.nan)
        feat['ret_open30']    = np.float32(np.nansum(r[idx_open])) if cut > 0 else np.float32(np.nan)
        feat['ret_close30']   = np.float32(np.nansum(r[idx_close])) if cut > 0 else np.float32(np.nan)
        feat['ret_big']       = np.float32(np.nansum(r[big_mask])) if big_mask.any() else np.float32(np.nan)

        # 波动率与下行波动占比
        feat['vol_5m']   = std_np32(r)
        feat['vol_open'] = std_np32(r[idx_open]) if cut > 0 else np.float32(np.nan)
        feat['vol_close'] = std_np32(r[idx_close]) if cut > 0 else np.float32(np.nan)
        feat['down_vol_ratio_all']   = downside_vol_ratio_np32(r)
        feat['down_vol_ratio_open']  = downside_vol_ratio_np32(r[idx_open]) if cut > 0 else np.float32(np.nan)
        feat['down_vol_ratio_close'] = downside_vol_ratio_np32(r[idx_close]) if cut > 0 else np.float32(np.nan)

        rows.append(feat)
        idxs.append(pd.Timestamp(dates_np[s]))

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows, index=pd.DatetimeIndex(idxs, name='date')).sort_index()
    out = out.replace([np.inf, -np.inf], np.nan)
    for c in out.columns:
        out[c] = out[c].astype('float32')
    return out

# -----------------------------
# I/O：按股票批次流式读取
# -----------------------------
def _iter_stock_batches(parquet_path, batch_stock_size=BATCH_STOCK_SIZE_DEFAULT, needed_cols=None):
    dataset = ds.dataset(parquet_path, format="parquet")
    cols = ['order_book_id', 'datetime'] + (needed_cols or NEEDED_COLS)
    tb_codes = dataset.to_table(columns=['order_book_id'])
    codes = pc.unique(tb_codes['order_book_id']).to_pylist()
    codes = [c for c in codes if c is not None]
    codes.sort()
    total = len(codes)
    for i in range(0, total, batch_stock_size):
        sub_codes = codes[i:i+batch_stock_size]
        filt = pc.field('order_book_id').isin(pa.array(sub_codes))
        table = dataset.to_table(filter=filt, columns=cols, use_threads=True)
        df = table.to_pandas()
        for col in (needed_cols or NEEDED_COLS):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
        df.set_index(['order_book_id', 'datetime'], inplace=True)
        df.sort_index(inplace=True)
        yield df, (i, min(i + batch_stock_size, total), total)

# -----------------------------
# 批任务：在子进程内处理一批股票
# -----------------------------
def _process_batch(df_batch: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    result: Dict[str, pd.DataFrame] = {}
    for stk, sub in df_batch.groupby(level=0, sort=False):
        sub_df = sub.reset_index(level=0, drop=True)
        f = _build_daily_hf_features(sub_df)
        if not f.empty:
            result[stk] = f
    return result

# -----------------------------
# 主流程：并行（以“批”为任务），阻塞等待收集结果
# -----------------------------
def build_hf_features_streaming_with_progress(parquet_path, num_workers, batch_stock_size=BATCH_STOCK_SIZE_DEFAULT, max_inflight=MAX_INFLIGHT_DEFAULT):
    dataset = ds.dataset(parquet_path, format="parquet")
    tb_codes = dataset.to_table(columns=['order_book_id'])
    codes = pc.unique(tb_codes['order_book_id']).to_pylist()
    codes = [c for c in codes if c is not None]
    total_codes = len(codes)

    task_pbar = tqdm(total=total_codes, desc="生成当日高频特征(股票级)", unit="stk")

    # 预热 numba（主进程）
    _ = mdd_np32(np.array([0.01, -0.02, 0.03], dtype=np.float32))

    stock_daily_hf: Dict[str, pd.DataFrame] = {}
    inflight = []

    with PoolExec(max_workers=num_workers) as ex:
        for df_batch, (l, r, tot) in _iter_stock_batches(parquet_path, batch_stock_size, NEEDED_COLS):
            # 限制在途批次数
            while len(inflight) >= max_inflight:
                done_set, _ = wait(inflight, return_when=FIRST_COMPLETED)
                for fut in list(done_set):
                    inflight.remove(fut)
                    res = fut.result()
                    task_pbar.update(len(res))
                    stock_daily_hf.update(res)

            fut = ex.submit(_process_batch, df_batch)
            inflight.append(fut)

        # 收尾
        while inflight:
            done_set, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in list(done_set):
                inflight.remove(fut)
                res = fut.result()
                task_pbar.update(len(res))
                stock_daily_hf.update(res)

    task_pbar.close()
    return stock_daily_hf

# -----------------------------
# 主函数：写 HDF5 + Scaler（float32）
# -----------------------------
def main():
    print("加载 5min 数据（流式，按股票批次）...")

    # 交易日与周五采样
    cal = load_calendar(CFG.trading_day_file)
    fridays = weekly_fridays(cal)
    fridays = fridays[(fridays >= pd.Timestamp(CFG.start_date)) & (fridays <= pd.Timestamp(CFG.end_date))]

    # 并行计算当日高频特征
    stock_daily_hf = build_hf_features_streaming_with_progress(
        parquet_path=CFG.price_5m_file,
        num_workers=CFG.num_workers,                                           # 建议≈物理核
        batch_stock_size=getattr(CFG, 'batch_stock_size', BATCH_STOCK_SIZE_DEFAULT),
        max_inflight=getattr(CFG, 'max_inflight_batches', MAX_INFLIGHT_DEFAULT),
    )

    if not stock_daily_hf:
        raise RuntimeError("未能从 5min 数据生成任何当日高频特征，请检查数据完整性。")

    # 列顺序
    any_stock = next(iter(stock_daily_hf))
    hf_cols = stock_daily_hf[any_stock].columns.tolist()

    # 写 HDF5（float32）
    out_h5 = CFG.processed_dir / "features_5m.h5"
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    str_dt = h5py.string_dtype(encoding='utf-8')

    with h5py.File(out_h5, "w") as h5f:
        h5f.attrs['factor_cols_hf'] = np.asarray(hf_cols, dtype=str_dt)

        for date_idx, d in enumerate(tqdm(fridays, desc="写入 5min 特征仓")):
            feats_all: List[np.ndarray] = []
            stk_list: List[str] = []

            for stk, df_day in stock_daily_hf.items():
                df_win = df_day[df_day.index <= d].tail(CFG.hf_window)
                if len(df_win) < CFG.hf_window:
                    continue
                df_win = df_win.reindex(columns=hf_cols)
                vals = df_win.values.astype(np.float32, copy=False)
                if not np.isfinite(vals).any() or np.isnan(vals).all():
                    continue
                vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
                feats_all.append(vals)
                stk_list.append(stk)

            if not feats_all:
                continue

            feats_arr = np.stack(feats_all, axis=0).astype(np.float32, copy=False)
            feats_arr = mad_clip(feats_arr).astype(np.float32, copy=False)
            feats_arr = np.nan_to_num(feats_arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

            g = h5f.create_group(f"date_{date_idx}")
            g.attrs['date'] = d.strftime('%Y-%m-%d')
            g.create_dataset("stocks", data=np.asarray(stk_list, dtype=str_dt))
            g.create_dataset("factor_hf", data=feats_arr, compression="gzip")

    # 拟合 scaler（float32）
    samples = []
    with h5py.File(out_h5, "r") as h5f:
        for k in h5f.keys():
            if 'factor_hf' in h5f[k]:
                samples.append(h5f[k]['factor_hf'][:].astype(np.float32, copy=False))
    if not samples:
        raise RuntimeError("未生成 5min 特征样本，无法拟合 scaler。")

    all_arr = np.concatenate(samples, axis=0).astype(np.float32, copy=False)
    scaler = Scaler()
    scaler.fit(all_arr)
    pkl_dump(scaler, CFG.processed_dir / "scaler_5m.pkl")

    print(f"5min 聚合特征完成：C_h={all_arr.shape[-1]}, T_h={all_arr.shape[1]}")
    print(f"保存 -> {out_h5}; 标准化器 -> {CFG.processed_dir / 'scaler_5m.pkl'}")

if __name__ == "__main__":
    main()