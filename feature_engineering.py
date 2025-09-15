# -*- coding: utf-8 -*-
"""
feature_engineering.py

功能概述
- 量价特征：基于日线行情计算多组量价因子，按“每周最后一个交易日（近似周五）”采样，将 [N, T, C] 写入 H5；
- 基本面特征：读取基本面 TTM/MRQ 等字段，前向填充到交易日，计算派生特征；在采样日做“行业内 z 标准化”，打印覆盖率与样例；
- 稳健性修复：
  1) pct_change 显式 fill_method=None，消除 pandas FutureWarning；
  2) 行业内 z 标准化逐日期逐行业显式计算，避免 groupby.apply 相关告警与多级索引错误；对常量/全 NaN 组稳健返回 NaN；可选 1%/99% 分位截尾；
  3) 写盘前对数组做 MAD 裁剪与 NaN/Inf→0 处理，避免训练端崩溃。

与 config.py 的兼容
- 仅使用以下已存在字段：price_day_file、funda_day_file、trading_day_file、industry_map_file、processed_dir、daily_window。
- H5 输出路径固定为 CFG.processed_dir / "features_daily.h5"（即 config 里已有 processed_dir 下），不新增 CFG 字段。

运行
- python feature_engineering.py
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

from config import CFG
from utils import load_calendar, load_industry_map, ensure_multiindex_price, mad_clip

# ========== 告警静默 ==========
warnings.filterwarnings("ignore", message="The default fill_method='ffill' in SeriesGroupBy.pct_change")
warnings.filterwarnings("ignore", category=DeprecationWarning, message="DataFrameGroupBy.apply operated on the grouping columns")

# ========== 常量 ==========
EPS = 1e-12
_TRADING_DAYS_PER_YEAR = 252
_TRADING_DAYS_PER_QUARTER = 63
_COVERAGE_THRESHOLD = 0.60  # 基本面横截面覆盖率筛选

# 行业内 z 前建议截尾的重尾列
_HEAVY_TAIL_COLS = [
    "inv_pb", "inv_ps", "ev_to_sales",
    "cfo_to_profit", "cfo_to_sales", "accrual_proxy",
    "liabilities_to_ev",
    "d_inv_pb_q", "d_cfo_to_profit_q", "d_liabilities_to_ev_q",
]

# ========== 工具 ==========
def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / (b.replace(0, np.nan) + EPS)

def _winsorize_inplace(gdf: pd.DataFrame, cols: List[str], lq=0.01, uq=0.99):
    for c in cols:
        if c in gdf.columns:
            s = gdf[c]
            lo, hi = s.quantile(lq), s.quantile(uq)
            gdf[c] = s.clip(lower=lo, upper=hi)

def forward_fill_to_trading_days(df: pd.DataFrame, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    """
    将 MultiIndex(order_book_id, datetime) 的不规则披露数据对齐到交易日日历，并按股票前向填充。
    """
    sids = df.index.get_level_values(0).unique()
    full_index = pd.MultiIndex.from_product([sids, trading_days], names=["order_book_id", "datetime"])
    df2 = df.reindex(full_index).sort_index()
    df2 = df2.groupby(level=0).ffill()
    return df2

# ========== 量价基础函数 ==========
def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=1).mean()

def _std_rolling(s: pd.Series, win: int) -> pd.Series:
    return s.rolling(win, min_periods=1).std()

def _mean_rolling(s: pd.Series, win: int) -> pd.Series:
    return s.rolling(win, min_periods=1).mean()

def _max_rolling(s: pd.Series, win: int) -> pd.Series:
    return s.rolling(win, min_periods=1).max()

def _min_rolling(s: pd.Series, win: int) -> pd.Series:
    return s.rolling(win, min_periods=1).min()

def _returns(close: pd.Series) -> pd.Series:
    return close.pct_change()

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

def _kurtosis(series: pd.Series, win: int) -> pd.Series:
    return series.rolling(win, min_periods=1).kurt()

def _skew(series: pd.Series, win: int) -> pd.Series:
    return series.rolling(win, min_periods=1).skew()

def _corr(a: pd.Series, b: pd.Series, win: int) -> pd.Series:
    return a.rolling(win, min_periods=2).corr(b)

def _zscore(s: pd.Series, win: int) -> pd.Series:
    m = _mean_rolling(s, win); sd = _std_rolling(s, win)
    return (s - m) / (sd + EPS)

# ========== 量价因子（单股票） ==========
def calc_factors_one_stock_full(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    O, H, L, C = df['open'], df['high'], df['low'], df['close']
    V = df.get('volume', pd.Series(index=df.index, dtype='float64'))
    NT = df.get('num_trades', pd.Series(index=df.index, dtype='float64'))
    TT = df.get('total_turnover', pd.Series(index=df.index, dtype='float64'))

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

    boll_ma, _, _, boll_w, boll_pctb = _boll(C, 20, 2.0)
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

# ========== 基本面派生特征 ==========
def compute_funda_features(raw_ff: pd.DataFrame) -> pd.DataFrame:
    """
    输入：前向填充到交易日的基本面长表（MultiIndex: order_book_id, datetime）
    输出：加入派生列后的 DataFrame
    """
    df = raw_ff.replace([np.inf, -np.inf], np.nan).copy()

    # 价值/EV
    df["inv_pb"] = 1.0 / (df["pb_ratio_lf"].replace(0, np.nan))
    df["inv_ps"] = 1.0 / (df["ps_ratio_ttm"].replace(0, np.nan))
    df["ev_to_sales"] = df["ev_ttm"] / (df["operating_revenue_ttm_0"].abs() + EPS)

    # 盈利/成长
    df["profit_margin_ttm"] = df["net_profit_parent_company_ttm_0"] / (df["operating_revenue_ttm_0"].abs() + EPS)
    df["revenue_yoy_ttm"] = df.groupby(level=0)["operating_revenue_ttm_0"].pct_change(
        _TRADING_DAYS_PER_YEAR, fill_method=None
    )
    df["margin_change_qoq"] = df.groupby(level=0)["profit_margin_ttm"].diff(_TRADING_DAYS_PER_QUARTER)

    # 质量
    df["cfo_to_profit"] = df["cash_flow_from_operating_activities_ttm_0"] / (
        df["net_profit_parent_company_ttm_0"].replace(0, np.nan)
    )
    df["cfo_to_sales"] = df["cash_flow_from_operating_activities_ttm_0"] / (df["operating_revenue_ttm_0"].abs() + EPS)
    df["accrual_proxy"] = 1.0 - df["cfo_to_profit"]

    # 杠杆
    df["liabilities_to_ev"] = df["total_liabilities_mrq_0"] / (df["ev_ttm"].abs() + EPS)

    # 创新
    df["rd_intensity"] = df["r_n_d"] / (df["operating_revenue_ttm_0"].abs() + EPS)

    # 季度差
    df["d_inv_pb_q"] = df.groupby(level=0)["inv_pb"].diff(_TRADING_DAYS_PER_QUARTER)
    df["d_profit_margin_q"] = df.groupby(level=0)["profit_margin_ttm"].diff(_TRADING_DAYS_PER_QUARTER)
    df["d_cfo_to_profit_q"] = df.groupby(level=0)["cfo_to_profit"].diff(_TRADING_DAYS_PER_QUARTER)
    df["d_liabilities_to_ev_q"] = df.groupby(level=0)["liabilities_to_ev"].diff(_TRADING_DAYS_PER_QUARTER)

    # 掩码
    df["mask_loss"] = (df["net_profit_parent_company_ttm_0"] <= 0).astype("float32")
    df["mask_cfo_neg"] = (df["cash_flow_from_operating_activities_ttm_0"] <= 0).astype("float32")

    return df

def _industry_zscore_per_date(
    df_vals: pd.DataFrame,
    ind_map: Dict[str, int],
    dates: List[pd.Timestamp],
    winsorize_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    行业内 z 标准化（逐日期、逐行业），对小样本/全缺失稳健处理，噪声更少。
    """
    feat_cols = list(df_vals.columns)
    # 先做一次全局清理，降低 pandas 内部 nanops 的警告概率
    clean = df_vals.replace([np.inf, -np.inf], np.nan).copy()
    # 对所有列做一次轻度分位裁剪（全市场层面），避免极端值对均值/方差的影响
    for c in feat_cols:
        if clean[c].dtype.kind in "fc":
            lo, hi = clean[c].quantile(0.001), clean[c].quantile(0.999)
            clean[c] = clean[c].clip(lower=lo, upper=hi)

    out_list: List[pd.DataFrame] = []

    for dt in dates:
        try:
            x_dt = clean.xs(dt, level="datetime", drop_level=False)
        except KeyError:
            continue
        if x_dt.empty:
            continue

        sids = x_dt.index.get_level_values(0).astype(str)
        inds = pd.Series([ind_map.get(s, np.nan) for s in sids], index=x_dt.index)

        z_frames: List[pd.DataFrame] = []
        # 按行业切片
        for gid, idx in inds.groupby(inds).groups.items():
            gnum = x_dt.loc[idx, feat_cols].copy()

            # 有效样本数不足直接返回 NaN，避免内部 var 计算警告
            valid_count = gnum.notna().sum(axis=0).max()
            if valid_count is None or valid_count < 2:
                z_frames.append(pd.DataFrame(index=gnum.index, columns=feat_cols, dtype="float32"))
                continue

            # 组内 winsorize（只对给定列）
            if winsorize_cols:
                for col in winsorize_cols:
                    if col in gnum.columns and gnum[col].dtype.kind in "fc":
                        s = gnum[col]
                        lo, hi = s.quantile(0.01), s.quantile(0.99)
                        gnum[col] = s.clip(lower=lo, upper=hi)

            mean = gnum.mean(axis=0, skipna=True)
            std = gnum.std(axis=0, ddof=0, skipna=True)
            std = std.where(std > 0, np.nan)

            z = (gnum - mean) / std
            z_frames.append(z)

        if z_frames:
            z_dt = pd.concat(z_frames, axis=0).reindex(index=x_dt.index, columns=feat_cols)
        else:
            # 全市场 fallback
            gnum = x_dt[feat_cols]
            mean = gnum.mean(axis=0, skipna=True)
            std = gnum.std(axis=0, ddof=0, skipna=True)
            std = std.where(std > 0, np.nan)
            z_dt = (gnum - mean) / std

        out_list.append(z_dt)

    if not out_list:
        return pd.DataFrame(index=df_vals.index, columns=df_vals.columns, dtype="float32")

    res = pd.concat(out_list, axis=0).sort_index()
    res = res.reindex(index=df_vals.index, columns=df_vals.columns)
    return res.astype("float32")

# ========== 数据读取 ==========
def load_price_day() -> pd.DataFrame:
    df = pd.read_parquet(CFG.price_day_file)
    df = ensure_multiindex_price(df)  # -> MultiIndex(order_book_id, datetime)
    return df

def load_funda_parquet() -> pd.DataFrame:
    """
    使用 CFG.funda_day_file 指向的 parquet（MultiIndex 规范化）。
    """
    path = CFG.funda_day_file
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.MultiIndex):
        df.index = pd.MultiIndex.from_tuples(df.index, names=["order_book_id", "datetime"])
    dt = df.index.get_level_values(1)
    if not np.issubdtype(dt.dtype, np.datetime64):
        df.index = pd.MultiIndex.from_tuples([(sid, pd.to_datetime(x)) for sid, x in df.index],
                                             names=["order_book_id", "datetime"])
    return df.sort_index()

# ========== 基本面横截面构建 ==========
def build_fundamental_cross_section(
    trading_days: pd.DatetimeIndex,
    sample_days: pd.DatetimeIndex,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    返回：
    - funda_cs: 采样日横截面（行业内 z 后，拼接掩码列）；索引 MultiIndex(sid, datetime)
    - used_cols: 实际保留的特征列名（覆盖率过滤后）
    """
    # 读取与交易日对齐
    raw = load_funda_parquet()
    raw = raw[(raw.index.get_level_values("datetime") >= trading_days.min()) &
              (raw.index.get_level_values("datetime") <= trading_days.max())]
    raw_ff = forward_fill_to_trading_days(raw, trading_days)

    # 派生
    feats = compute_funda_features(raw_ff)

    # 采样日截面
    feats_cs = feats[feats.index.get_level_values("datetime").isin(sample_days)]

    # 行业内 z
    stock2ind = load_industry_map(CFG.industry_map_file)
    base_cols = [
        "inv_pb", "inv_ps", "ev_to_sales",
        "profit_margin_ttm", "revenue_yoy_ttm", "margin_change_qoq",
        "cfo_to_profit", "cfo_to_sales", "accrual_proxy",
        "liabilities_to_ev", "rd_intensity",
        "d_inv_pb_q", "d_profit_margin_q", "d_cfo_to_profit_q", "d_liabilities_to_ev_q",
    ]
    mask_cols = ["mask_loss", "mask_cfo_neg"]

    df_base = feats_cs[base_cols].copy()
    df_z = _industry_zscore_per_date(df_base, stock2ind, list(sample_days), winsorize_cols=_HEAVY_TAIL_COLS)

    # 覆盖率过滤
    keep_cols = []
    for c in df_z.columns:
        ratio = 1.0 - df_z[c].isna().mean()
        if ratio >= _COVERAGE_THRESHOLD:
            keep_cols.append(c)

    funda_cs = pd.concat([df_z[keep_cols], feats_cs[mask_cols]], axis=1).astype("float32")
    return funda_cs, keep_cols + mask_cols

# ========== 量价特征全市场计算 ==========
def build_price_factors(price_day: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """
    计算所有股票的量价因子时序（可按需替换为并行）。返回 dict 与列名列表。
    """
    stock_feat: Dict[str, pd.DataFrame] = {}
    sids = price_day.index.get_level_values(0).unique().tolist()
    for sid in tqdm(sids, desc="计算量价因子(串行)"):
        hist = price_day.loc[sid]
        if isinstance(hist, pd.Series):
            hist = hist.to_frame().T
        hist = hist.sort_index()
        fct = calc_factors_one_stock_full(hist)
        stock_feat[sid] = fct

    if not stock_feat:
        return {}, []

    factor_cols = next(iter(stock_feat.values())).columns.tolist()
    return stock_feat, factor_cols

# ========== 主流程：基本面 + 量价写 H5 ==========
def main():
    print("[feature_engineering] start")

    # 1) 交易日与采样周五（每周最后一个交易日）
    cal = load_calendar(CFG.trading_day_file)
    df_cal = pd.DataFrame({"d": cal})
    df_cal["w"] = df_cal["d"].dt.to_period("W")
    sample_fridays = pd.DatetimeIndex(df_cal.groupby("w")["d"].last().values)

    # 2) 基本面横截面（行业内 z 后）
    funda_cs, funda_cols = build_fundamental_cross_section(trading_days=cal, sample_days=sample_fridays)
    print(f"[funda] cross-section shape={funda_cs.shape}, cols={len(funda_cols)}")
    # 打印覆盖率与样例（仅日志用途）
    if not funda_cs.empty:
        print("[funda] head sample:")
        try:
            print(funda_cs.groupby(level=1).head(3).head(10))
        except Exception:
            print("样例打印失败，但不影响流程")

    # 3) 读取日线行情并限定到交易日范围
    price_day = load_price_day()
    price_day = price_day[(price_day.index.get_level_values("datetime") >= cal.min()) &
                          (price_day.index.get_level_values("datetime") <= cal.max())]

    # 4) 计算全市场量价特征
    stock_feat, factor_cols = build_price_factors(price_day)
    if not stock_feat:
        print("[price] empty features, stop")
        return

    # 5) 写入 H5（按采样日组织组）
    out_path = Path(CFG.processed_dir) / "features_daily.h5"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    str_dt = h5py.string_dtype(encoding="utf-8")
    if out_path.exists():
        # 为避免旧文件干扰，这里覆盖写；若你希望增量写，可改写为 "a" 模式并检查重名组。
        out_path.unlink()

    T = int(CFG.daily_window)
    with h5py.File(out_path, "w") as h5f:
        # 记录列名元数据
        h5f.attrs["factor_cols"] = np.asarray(factor_cols, dtype=str_dt)

        group_idx = 0
        write_cnt = 0
        for d in tqdm(sample_fridays, desc="写入H5(周五采样)"):
            feats_all = []
            stk_list = []

            for stk, fct_full in stock_feat.items():
                fct_hist = fct_full[fct_full.index <= d]
                if len(fct_hist) < T:
                    continue
                fct_win = fct_hist.tail(T).reindex(columns=factor_cols)
                vals = fct_win.values
                # 数值清理
                vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                feats_all.append(vals)
                stk_list.append(stk)

            if not feats_all:
                continue

            feats_arr = np.stack(feats_all, axis=0)  # [N, T, C]
            feats_arr = mad_clip(feats_arr)  # MAD 裁剪（来自 utils）
            feats_arr = np.nan_to_num(feats_arr, nan=0.0, posinf=0.0, neginf=0.0)

            g = h5f.create_group(f"date_{group_idx}")
            g.attrs["date"] = d.strftime("%Y-%m-%d")
            g.create_dataset("stocks", data=np.asarray(stk_list, dtype=str_dt))

            N, TT, Cc = feats_arr.shape
            chunk0 = int(min(256, N))
            g.create_dataset("factor", data=feats_arr, compression="lzf", chunks=(chunk0, TT, Cc))

            group_idx += 1
            write_cnt += 1

    print(f"[price] H5 written groups={write_cnt}, path={out_path}")
    print("[feature_engineering] done")

if __name__ == "__main__":
    main()