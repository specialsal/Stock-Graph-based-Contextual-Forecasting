# coding: utf-8
"""
增量版: 生成/追加 features_daily.h5

提速策略：
- 只对“缺失的周五采样日”进行追加；
- 仅计算满足这些采样日所需的“局部历史片段”: [history_start, end_date]；
- 线程并行对该片段内的每只股票计算全量因子（相对片段），采样时做窗口切片；
- 一次性 mad_clip，避免重复标准化；不保存全局 Scaler。

改进点（边界修订覆盖）：
- 从“最后一个已写入周五”起重算该周五并覆盖，避免边界遗漏/修订差异；
- 覆盖策略：写入前若目标日期已存在，先删除该组再写入（快速、无需重排编号）。

注意：
- 需要 utils.read_h5_meta / list_missing_fridays / get_required_history_start / ensure_multiindex_price 等工具函数。
- 最长回溯窗口为 120（与当前因子定义一致）；若后续新增更长窗口，请同步更新 CFG.max_lookback。
- 本版本在原有技术因子的基础上，额外接入日度基本面因子（来自 stock_fundamental_day.parquet），
  并将 market_cap_2 以对数形式（log_mktcap2）作为因子使用。
"""

import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import CFG
from utils import (
    mad_clip, weekly_fridays, load_calendar,
    read_h5_meta, list_missing_fridays, get_required_history_start,
    ensure_multiindex_price, SlidingWindowCache
)

# ===== 数值稳定工具 =====
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

# ===== 因子计算（与原版一致，仅技术因子） =====
def calc_factors_one_stock_full(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入 df 需至少包含行情列：open, high, low, close, volume, num_trades, total_turnover
    该函数只用这些行情列计算技术因子；不会直接使用基本面列。
    基本面将由外层在时间轴上 join 进来。
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

    out['vol_ret_corr20_lag1']  = _corr(V.pct_change(), ret.shift(1), 20)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out

# ===== 并行包装（技术因子 + 基本面因子拼接） =====
def _compute_one(stock: str,
                 price_all: pd.DataFrame,
                 fund_all: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
    """
    对单只股票，在给定的局部行情片段 price_all 和基本面片段 fund_all 上：
    - 先用 price_all 计算技术因子（calc_factors_one_stock_full）
    - 再将对齐到日期的基本面列 join 进来
    返回 index=日期 的因子 DataFrame（技术 + 基本面）
    """
    try:
        # 行情历史
        hist_price = price_all.loc[stock]
        if isinstance(hist_price, pd.Series):
            hist_price = hist_price.to_frame().T
        hist_price = hist_price.sort_index()

        # 对应股票的基本面历史（可能缺失）
        if stock in fund_all.index.get_level_values(0):
            hist_fund = fund_all.loc[stock]
            if isinstance(hist_fund, pd.Series):
                hist_fund = hist_fund.to_frame().T
            hist_fund = hist_fund.sort_index()
            # 按日期对齐，左连接使得所有行情日期保留，基本面缺失为 NaN
            hist_merged = hist_price.join(hist_fund, how="left")
        else:
            hist_merged = hist_price.copy()

        # 只用行情部分计算技术因子
        fct_tech = calc_factors_one_stock_full(hist_merged)

        # 将基本面列 join 进因子表（按日期对齐）
        fund_cols_here = [c for c in fund_all.columns if c in hist_merged.columns]
        if fund_cols_here:
            fct_all = fct_tech.join(hist_merged[fund_cols_here], how="left")
        else:
            fct_all = fct_tech

        return stock, fct_all
    except Exception:
        return stock, pd.DataFrame()

def enforce_factor_cols_consistency(stock_feat: Dict[str, pd.DataFrame], existed_cols: Optional[List[str]]) -> List[str]:
    if existed_cols is not None and len(existed_cols) > 0:
        return list(existed_cols)
    if len(stock_feat) == 0:
        raise RuntimeError("无可用的股票因子用于推断列名。")
    sample_stock = next(iter(stock_feat.keys()))
    return stock_feat[sample_stock].columns.tolist()

# ===== 主流程（增量 + 局部片段 + 覆盖最后一日） =====
def main():
    # 1) 读取最新的日频行情数据并规范索引
    day_df = pd.read_parquet(CFG.price_day_file)
    day_df = ensure_multiindex_price(day_df)

    stocks = day_df.index.get_level_values(0).unique().tolist()

    # 1.1) 读取基本面日频数据，并与行情索引对齐
    fund_df = pd.read_parquet(CFG.price_fundamental_file)

    if not isinstance(fund_df.index, pd.MultiIndex) or fund_df.index.nlevels != 2:
        # 假设有显式列 order_book_id, date
        fund_df = fund_df.set_index(["order_book_id", "date"]).sort_index()

    # 将 MultiIndex 第二级名称统一为 'datetime'，与 day_df 一致
    names = list(fund_df.index.names)
    if names != ["order_book_id", "datetime"]:
        try:
            fund_df.index = fund_df.index.set_names(["order_book_id", "datetime"])
        except Exception:
            pass
    fund_df = fund_df.sort_index()

    # 对 market_cap_2 做 log 变换，得到对数流通市值因子
    if "market_cap_2" in fund_df.columns:
        mc = fund_df["market_cap_2"].astype(float)
        # clip 以避免 log(0) 或 log(非常小的负数/噪声)
        fund_df["log_mktcap2"] = np.log(np.clip(mc, 1e4, None))
    else:
        raise ValueError("stock_fundamental_day.parquet 缺少列 'market_cap_2'")

    # 只保留我们需要的 8 个基本面因子（其中市值使用 log 变换后的 log_mktcap2）
    fund_cols = [
        "pe_ratio_ttm",
        "book_to_market_ratio_ttm",
        "ps_ratio_ttm",
        "log_mktcap2",                   # 对数流通市值
        "dividend_yield_ttm",
        "cash_flow_per_share_ttm",
        "inc_revenue_ttm",
        "net_profit_growth_ratio_ttm",
    ]
    missing_fund = [c for c in fund_cols if c not in fund_df.columns]
    if missing_fund:
        raise ValueError(f"fundamental 数据缺少必要列: {missing_fund}")
    fund_df = fund_df[fund_cols].copy()

    # 2) 已写入的 H5 元信息
    h5_path = Path(CFG.feat_file)
    next_idx, written_dates, existed_cols = read_h5_meta(h5_path)
    written_dates_idx = pd.DatetimeIndex(sorted(written_dates)) if written_dates else pd.DatetimeIndex([])
    last_written = written_dates_idx.max() if len(written_dates_idx) > 0 else None

    # 3) 缺失的周五（加入 last_written 用于覆盖修订）
    missing_fridays = list_missing_fridays(Path(CFG.trading_day_file), CFG.start_date, CFG.end_date, written_dates)
    missing_fridays = list(missing_fridays)  # 确保是 list，便于 append
    if last_written is not None and last_written not in missing_fridays:
        missing_fridays.append(last_written)
    # 最终转为 DatetimeIndex，供下游函数（需要 .min()/.max()）使用
    missing_fridays = pd.DatetimeIndex(sorted(set(missing_fridays)))
    if len(missing_fridays) == 0:
        print("没有新增/需覆盖的周五采样日需要写入，增量构建完成。")
        return

    # 4) 局部历史片段的起点
    hist_start = get_required_history_start(missing_fridays, max_lookback=CFG.max_lookback)
    hist_end = pd.Timestamp(CFG.end_date)
    # 截取行情片段，仅用这一段计算全量因子
    print(f"局部历史片段: {hist_start.date()} ~ {hist_end.date()}, 读取中...")
    sliced = day_df[(day_df.index.get_level_values('datetime') >= hist_start) &
                    (day_df.index.get_level_values('datetime') <= hist_end)]
    print(f"读取局部行情片段: {sliced.index.get_level_values('datetime').min().date()} ~ {sliced.index.get_level_values('datetime').max().date()}, "
          f"股票数={sliced.index.get_level_values(0).nunique()}")

    # 对基本面也截取同一区间
    fund_sliced = fund_df[(fund_df.index.get_level_values('datetime') >= hist_start) &
                          (fund_df.index.get_level_values('datetime') <= hist_end)]

    # 5) 并行计算“局部片段”的全量因子缓存（技术 + 基本面）
    stock_feat: Dict[str, pd.DataFrame] = {}
    valid_stocks_price = set(sliced.index.get_level_values(0).unique())
    with ThreadPoolExecutor(max_workers=CFG.num_workers) as ex:
        futures = {
            ex.submit(_compute_one, s, sliced, fund_sliced): s
            for s in stocks if s in valid_stocks_price
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="并行计算因子(局部片段)"):
            s = futures[fut]
            stk, fct = fut.result()
            if not fct.empty:
                stock_feat[stk] = fct

    if len(stock_feat) == 0:
        print("局部片段内无可计算的股票因子，增量构建结束。")
        return

    # 6) 因子列名与历史一致（首次构建时会自动包含技术 + 基本面所有列）
    factor_cols = enforce_factor_cols_consistency(stock_feat, existed_cols)

    # 7) 逐个周五写入（只处理缺失/需覆盖的）
    str_dt = h5py.string_dtype(encoding='utf-8')
    mode = "a" if h5_path.exists() else "w"
    written_cnt_total = 0
    with h5py.File(h5_path, mode) as h5f:
        # 建立 date -> group_name 映射（用于覆盖删除）
        existing_groups: Dict[pd.Timestamp, str] = {}
        for k in h5f.keys():
            try:
                gd = h5f[k].attrs.get('date', None)
                if gd is not None:
                    existing_groups[pd.Timestamp(gd)] = k
            except Exception:
                continue

        if 'factor_cols' not in h5f.attrs:
            h5f.attrs['factor_cols'] = np.asarray(factor_cols, dtype=str_dt)

        group_idx = next_idx
        for d in tqdm(missing_fridays, desc="增量写入特征仓"):
            # 覆盖：如果已存在该日期组，先删除
            if d in existing_groups:
                try:
                    del h5f[existing_groups[d]]
                except Exception:
                    # 删除失败则忽略，后面追加会多一组；但 read_h5_meta 会以 attrs['date'] 来识别
                    pass

            feats_all = []
            stk_list  = []
            # 对所有股票在该日期切出窗口
            for stk, fct_full in stock_feat.items():
                fct_hist = fct_full[fct_full.index <= d]
                if len(fct_hist) < CFG.daily_window:
                    continue
                fct_win = fct_hist.tail(CFG.daily_window)
                # 对齐列顺序（技术 + 基本面）
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
            # 对所有因子统一做 MAD 裁剪和 NaN→0（包括基本面）
            feats_arr = mad_clip(feats_arr)
            feats_arr = np.nan_to_num(feats_arr, nan=0.0, posinf=0.0, neginf=0.0)

            g = h5f.create_group(f"date_{group_idx}")
            g.attrs['date'] = d.strftime('%Y-%m-%d')
            g.create_dataset("stocks", data=np.asarray(stk_list, dtype=str_dt))

            N, T, C = feats_arr.shape
            chunk0 = int(min(256, N))
            try:
                g.create_dataset(
                    "factor",
                    data=feats_arr,
                    compression="lzf",
                    chunks=(chunk0, T, C)
                )
            except Exception:
                g.create_dataset(
                    "factor",
                    data=feats_arr,
                    compression=None,
                    chunks=(chunk0, T, C)
                )
            written_cnt_total += 1
            group_idx += 1

    print(f"增量构建完成：新增/覆盖周五组数={written_cnt_total}，H5路径={h5_path}")

if __name__ == "__main__":
    main()