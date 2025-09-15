# -*- coding: utf-8 -*-
"""
feature_engineering.py

说明
- 生成量价特征与基本面特征的横截面，并在采样日写入 H5。
- 本版修复/增强：
  1) 行业内 z 标准化函数 _industry_zscore_per_date：不再使用 groupby.apply，避免弃用告警与多级索引错误；
     逐日期逐行业显式循环，稳健处理常量与全 NaN 组，支持重尾截尾；
     严格返回两级 MultiIndex [order_book_id, datetime]，与输入对齐。
  2) 同比计算 pct_change 显式 fill_method=None，消除 FutureWarning。
  3) 对 NaN/Inf 做安全替换与覆盖率检查，避免下游训练异常。

依赖
- config.CFG: 提供路径/窗口等配置
- utils: 日历/行业映射/读写工具等
"""

import os
import sys
import math
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

# 可按需静默部分历史告警
warnings.filterwarnings("ignore", message="The default fill_method='ffill' in SeriesGroupBy.pct_change")
warnings.filterwarnings("ignore", category=DeprecationWarning, message="DataFrameGroupBy.apply operated on the grouping columns")

# 工程内依赖（请确保这些模块在你的工程内存在且路径正确）
from config import CFG
from utils import (
    ensure_multiindex_price,
    load_calendar,
    load_industry_map,
    mad_clip,
)

# =========================
# 常量与工具
# =========================
EPS = 1e-12
_TRADING_DAYS_PER_YEAR = 252
_TRADING_DAYS_PER_QUARTER = 63
_COVERAGE_THRESHOLD = 0.60  # 横截面覆盖率门槛

_HEAVY_TAIL_COLS = [
    # 量价与基本面易重尾的列名，可按需扩展；仅用于行业内 z 前的 winsorize
    "inv_pb", "inv_ps", "ev_to_sales",
    "cfo_to_profit", "cfo_to_sales", "accrual_proxy",
    "liabilities_to_ev",
    "d_inv_pb_q", "d_cfo_to_profit_q", "d_liabilities_to_ev_q",
]

def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / (b.replace(0, np.nan) + EPS)

def _winsorize_by_quantile(s: pd.Series, lower=0.01, upper=0.99) -> pd.Series:
    if s.dtype.kind not in "fc":  # 非浮点/复数直接返回
        return s
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lower=lo, upper=hi)

# =========================
# 量价基础函数（与原工程保持一致口径）
# =========================
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

# =========================
# 量价因子（单股票）
# =========================
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

# =========================
# 基本面派生特征
# 输入：MultiIndex(order_book_id, datetime)
# 列：基础字段 pb_ratio_lf, ps_ratio_ttm, operating_revenue_ttm_0, net_profit_parent_company_ttm_0,
#     cash_flow_from_operating_activities_ttm_0, total_liabilities_mrq_0, ev_ttm, r_n_d
# =========================
def compute_funda_features(raw_ff: pd.DataFrame) -> pd.DataFrame:
    df = raw_ff.copy()

    # 价值/EV
    df["inv_pb"] = 1.0 / (df["pb_ratio_lf"].replace(0, np.nan))
    df["inv_ps"] = 1.0 / (df["ps_ratio_ttm"].replace(0, np.nan))
    df["ev_to_sales"] = df["ev_ttm"] / (df["operating_revenue_ttm_0"].abs() + EPS)

    # 盈利/成长
    df["profit_margin_ttm"] = df["net_profit_parent_company_ttm_0"] / (df["operating_revenue_ttm_0"].abs() + EPS)
    # 显式关闭前向填充，避免 pandas FutureWarning
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

    # 小幅变化（季度差）
    df["d_inv_pb_q"] = df.groupby(level=0)["inv_pb"].diff(_TRADING_DAYS_PER_QUARTER)
    df["d_profit_margin_q"] = df.groupby(level=0)["profit_margin_ttm"].diff(_TRADING_DAYS_PER_QUARTER)
    df["d_cfo_to_profit_q"] = df.groupby(level=0)["cfo_to_profit"].diff(_TRADING_DAYS_PER_QUARTER)
    df["d_liabilities_to_ev_q"] = df.groupby(level=0)["liabilities_to_ev"].diff(_TRADING_DAYS_PER_QUARTER)

    # 二元指示
    df["mask_loss"] = (df["net_profit_parent_company_ttm_0"] <= 0).astype("float32")
    df["mask_cfo_neg"] = (df["cash_flow_from_operating_activities_ttm_0"] <= 0).astype("float32")

    return df

# =========================
# 行业内 z 标准化（逐日期、逐行业）
# 返回与输入 df_vals 同索引/列，dtype=float32
# =========================
def _industry_zscore_per_date(
    df_vals: pd.DataFrame,
    ind_map: Dict[str, int],
    dates: List[pd.Timestamp],
    winsorize_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    逐日期、逐行业计算横截面 z-score（不使用 groupby.apply，避免弃用告警；仅两级索引）。
    输入
    - df_vals: 宽表，MultiIndex(order_book_id, datetime)，列为特征名。建议仅包含采样日，加速。
    - ind_map: 股票 -> 行业ID 的 dict
    - dates: 需要计算的日期序列（采样日）
    - winsorize_cols: 分行业 1%/99% 分位截尾列名

    返回
    - 与 df_vals 同索引/列的 z 值 DataFrame（float32）
    """
    feat_cols = list(df_vals.columns)
    out_list: List[pd.DataFrame] = []

    for dt in dates:
        try:
            x_dt = df_vals.xs(dt, level="datetime", drop_level=False)
        except KeyError:
            continue
        if x_dt.empty:
            continue

        # 股票 -> 行业ID。维持两级索引对齐
        sids = x_dt.index.get_level_values(0).astype(str)
        inds = pd.Series([ind_map.get(s, np.nan) for s in sids], index=x_dt.index)

        z_pieces: List[pd.DataFrame] = []
        # 以 inds 的分组索引进行切片，避免把分组列并入 X
        for gid, g_idx in inds.groupby(inds).groups.items():
            gnum = x_dt.loc[g_idx, feat_cols].copy()

            # 可选 winsorize
            if winsorize_cols:
                for col in winsorize_cols:
                    if col in gnum.columns:
                        s = gnum[col]
                        lo, hi = s.quantile(0.01), s.quantile(0.99)
                        gnum[col] = s.clip(lower=lo, upper=hi)

            # 组内均值/方差（常量或全缺失 -> std=0 或 NaN）
            mean = gnum.mean(axis=0, skipna=True)
            std = gnum.std(axis=0, ddof=0, skipna=True)
            std = std.where(std != 0, np.nan)

            z = (gnum - mean) / std
            z_pieces.append(z)

        if len(z_pieces) == 0:
            # 回退到全市场 z
            gnum = x_dt[feat_cols]
            mean = gnum.mean(axis=0, skipna=True)
            std = gnum.std(axis=0, ddof=0, skipna=True)
            std = std.where(std != 0, np.nan)
            z_dt = (gnum - mean) / std
        else:
            z_dt = pd.concat(z_pieces, axis=0)
            # 确保索引与 x_dt 一一对应（两级 MultiIndex），列只保留 feat_cols
            z_dt = z_dt.reindex(index=x_dt.index, columns=feat_cols)

        out_list.append(z_dt)

    if not out_list:
        # 返回空框架，但索引和列需与 df_vals 一致
        return pd.DataFrame(index=df_vals.index, columns=df_vals.columns, dtype="float32")

    # 逐日拼接，索引本就包含 datetime，保持两级 MultiIndex
    res = pd.concat(out_list, axis=0).sort_index()
    # 最终严格对齐原索引与列，避免出现多于两级的索引层导致 reindex 失败
    res = res.reindex(index=df_vals.index, columns=df_vals.columns)
    return res.astype("float32")

# =========================
# 数据读取
# =========================
def load_price_day() -> pd.DataFrame:
    df = pd.read_parquet(CFG.price_day_file)
    df = ensure_multiindex_price(df)  # -> MultiIndex(order_book_id, datetime)
    return df

def load_funda_parquet() -> pd.DataFrame:
    path = CFG.funda_parquet  # e.g., data/raw/funda_factors.parquet
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.MultiIndex):
        df.index = pd.MultiIndex.from_tuples(df.index, names=["order_book_id", "datetime"])
    # 规范 datetime
    dt = df.index.get_level_values(1)
    if not np.issubdtype(dt.dtype, np.datetime64):
        df.index = pd.MultiIndex.from_tuples([(sid, pd.to_datetime(x)) for sid, x in df.index],
                                             names=["order_book_id", "datetime"])
    return df.sort_index()

def forward_fill_to_trading_days(df: pd.DataFrame, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    sids = df.index.get_level_values(0).unique()
    full_index = pd.MultiIndex.from_product([sids, trading_days], names=["order_book_id", "datetime"])
    df2 = df.reindex(full_index).sort_index()
    df2 = df2.groupby(level=0).ffill()
    return df2

# =========================
# 构建基本面横截面（采样日）
# =========================
def build_fundamental_cross_section(
    trading_days: pd.DatetimeIndex,
    sample_days: pd.DatetimeIndex,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    返回：
    - funda_cs: 采样日横截面（行业内 z 后，拼接部分掩码）；索引 MultiIndex(sid, datetime)
    - used_cols: 实际保留的特征列名（覆盖率过滤后）
    """
    # 1) 读取基本面源数据并前向填充至交易日
    raw = load_funda_parquet()
    # 限定至交易日范围
    raw = raw[(raw.index.get_level_values("datetime") >= trading_days.min()) &
              (raw.index.get_level_values("datetime") <= trading_days.max())]
    raw_ff = forward_fill_to_trading_days(raw, trading_days)

    # 2) 计算派生特征
    feats = compute_funda_features(raw_ff)

    # 3) 取采样日
    feats_cs = feats[feats.index.get_level_values("datetime").isin(sample_days)]

    # 4) 行业内 z 标准化
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
    dates = list(sample_days)
    df_z = _industry_zscore_per_date(df_base, stock2ind, dates, winsorize_cols=_HEAVY_TAIL_COLS)

    # 5) 覆盖率筛选与合并掩码列
    keep_cols = []
    for c in df_z.columns:
        ratio = 1.0 - df_z[c].isna().mean()
        if ratio >= _COVERAGE_THRESHOLD:
            keep_cols.append(c)
    funda_cs = pd.concat([df_z[keep_cols], feats_cs[mask_cols]], axis=1).astype("float32")

    return funda_cs, keep_cols + mask_cols

# =========================
# 构建量价因子（可按需在主流程中启用）
# =========================
def build_price_factors(
    price_day: pd.DataFrame,
    trading_days: pd.DatetimeIndex,
    sample_days: pd.DatetimeIndex,
    daily_window: int,
) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """
    返回：
    - stock_feat: dict[sid] -> 全时序量价特征 DataFrame
    - factor_cols: 列名列表
    """
    # 计算所有股票的量价因子时序（可按需并行，这里保持简单串行以清晰）
    stock_feat: Dict[str, pd.DataFrame] = {}
    sids = price_day.index.get_level_values(0).unique().tolist()
    for sid in tqdm(sids, desc="计算量价因子(串行示例，可自行改为并行)"):
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

# =========================
# 主流程：示例将基本面横截面与量价窗口写入 H5
# 可按需仅启用基本面验证（将价格段注释掉）
# =========================
def main():
    print("[feature_engineering] start")

    # 读取交易日与周五采样
    cal = load_calendar(CFG.trading_day_file)
    if CFG.sample_start is not None:
        cal = cal[cal >= pd.Timestamp(CFG.sample_start)]
    if CFG.sample_end is not None:
        cal = cal[cal <= pd.Timestamp(CFG.sample_end)]

    # 采样日：每周最后一个交易日（近似周五）
    df_cal = pd.DataFrame({"d": cal})
    df_cal["w"] = df_cal["d"].dt.to_period("W")
    sample_fridays = pd.DatetimeIndex(df_cal.groupby("w")["d"].last().values)

    # 基本面横截面
    funda_cs, funda_cols = build_fundamental_cross_section(trading_days=cal, sample_days=sample_fridays)
    print(f"[funda] cross-section shape={funda_cs.shape}, cols={len(funda_cols)}")

    # 写入一个示例 Parquet（横截面），便于核查
    if CFG.debug_dump_funda_cs:
        os.makedirs(os.path.dirname(CFG.debug_dump_funda_cs), exist_ok=True)
        funda_cs.to_parquet(CFG.debug_dump_funda_cs)
        print(f"[funda] dumped cross-section to {CFG.debug_dump_funda_cs}")

    # 以下为量价+H5 写入的示意，如果当前仅验证基本面，可注释此块
    price_day = load_price_day()
    # 限定到全体交易日范围
    price_day = price_day[(price_day.index.get_level_values("datetime") >= cal.min()) &
                            (price_day.index.get_level_values("datetime") <= cal.max())]
    stock_feat, factor_cols = build_price_factors(price_day, cal, sample_fridays, CFG.daily_window)
    if not stock_feat:
        print("[price] empty features, skip H5")
        return

    # 写入 H5：以采样日为组，每组包含 [N, T, C]
    out_path = CFG.features_h5
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    str_dt = h5py.string_dtype(encoding="utf-8")
    if os.path.exists(out_path) and CFG.overwrite_h5:
        os.remove(out_path)

    with h5py.File(out_path, "w") as h5f:
        h5f.attrs["factor_cols"] = np.asarray(factor_cols, dtype=str_dt)

        group_idx = 0
        write_cnt = 0
        for d in tqdm(sample_fridays, desc="写入H5(周五采样)"):
            feats_all = []
            stk_list = []
            for stk, fct_full in stock_feat.items():
                fct_hist = fct_full[fct_full.index <= d]
                if len(fct_hist) < CFG.daily_window:
                    continue
                fct_win = fct_hist.tail(CFG.daily_window).reindex(columns=factor_cols)
                vals = fct_win.values
                vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                feats_all.append(vals)
                stk_list.append(stk)

            if not feats_all:
                continue

            feats_arr = np.stack(feats_all, axis=0)  # [N, T, C]
            feats_arr = mad_clip(feats_arr)
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