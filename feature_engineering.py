# coding: utf-8
"""
增量版: 生成/追加 features_daily.h5

在原有量价特征流水线上，新增“基本面派生特征”的构造与拼接（常数序列注入）：
- 从 data/raw/funda_factors.parquet 读取 8 个源字段（PIT、日频）；
- 交易日对齐并前向填充；
- 构造派生特征：
  价值/EV：inv_pb, inv_ps, ev_to_sales
  盈利/成长：profit_margin_ttm, revenue_yoy_ttm, margin_change_qoq
  质量：cfo_to_profit, cfo_to_sales, accrual_proxy
  杠杆：liabilities_to_ev
  创新：rd_intensity
  小幅变化：d_inv_pb_q, d_profit_margin_q, d_cfo_to_profit_q, d_liabilities_to_ev_q
  二元指示：mask_loss, mask_cfo_neg
- 对上述除掩码外的列进行“行业内 z 标准化”（逐周五横截面、逐行业），重尾列先 1%/99% 截尾；
- 覆盖率不足（<60%）的特征自动降级；
- 在每个目标周五，将基本面横截面值复制为长度=CFG.daily_window 的常数序列，并与量价窗口在通道维拼接；
- 其余增量、覆盖、写盘逻辑保持与原版一致。

依赖
- config.CFG：路径/窗口配置
- utils：mad_clip, weekly_fridays, load_calendar, read_h5_meta, list_missing_fridays,
         get_required_history_start, ensure_multiindex_price, SlidingWindowCache, load_industry_map
"""

import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

from config import CFG
from utils import (
    mad_clip, weekly_fridays, load_calendar,
    read_h5_meta, list_missing_fridays, get_required_history_start,
    ensure_multiindex_price, SlidingWindowCache, load_industry_map
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

# ====== 基本面：配置与工具 ======
# 行业内 z 的重尾列（先做 1%/99% 截尾）
_HEAVY_TAIL_COLS = [
    "inv_pb", "inv_ps", "ev_to_sales",
    "cfo_to_profit", "cfo_to_sales", "accrual_proxy",
    "liabilities_to_ev",
    "d_inv_pb_q", "d_cfo_to_profit_q", "d_liabilities_to_ev_q",
]
# 需要行业内 z 的列
_FUNDA_BASE_COLS = [
    "inv_pb", "inv_ps", "ev_to_sales",
    "profit_margin_ttm", "revenue_yoy_ttm", "margin_change_qoq",
    "cfo_to_profit", "cfo_to_sales", "accrual_proxy",
    "liabilities_to_ev", "rd_intensity",
    "d_inv_pb_q", "d_profit_margin_q", "d_cfo_to_profit_q", "d_liabilities_to_ev_q",
]
# 掩码列
_FUNDA_MASK_COLS = ["mask_loss", "mask_cfo_neg"]
# 覆盖率阈值
_COVERAGE_THRESHOLD = 0.60
# 同比 / 季差近似
_TRADING_DAYS_PER_YEAR = 252
_TRADING_DAYS_PER_QUARTER = 63

def _winsorize_by_quantile(s: pd.Series, lower=0.01, upper=0.99) -> pd.Series:
    if s.dtype.kind not in "fc":
        return s
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lower=lo, upper=hi)

def _industry_zscore_per_date(df_vals: pd.DataFrame, ind_map: Dict[str, int], dates: List[pd.Timestamp],
                              winsorize_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    逐日期、逐行业计算横截面 z-score。ind_map 为股票->行业ID 的 dict（来自 utils.load_industry_map）。
    """
    out = []
    for dt in dates:
        try:
            x = df_vals.xs(dt, level="datetime", drop_level=False)
        except KeyError:
            continue
        sids = x.index.get_level_values(0).astype(str)
        inds = pd.Series([ind_map.get(s, np.nan) for s in sids], index=x.index, name="_industry_id")
        tmp = pd.concat([x, inds], axis=1)

        # 重尾列分组 winsorize
        if winsorize_cols:
            for col in winsorize_cols:
                if col in tmp.columns:
                    tmp[col] = tmp.groupby("_industry_id")[col].transform(lambda s: _winsorize_by_quantile(s, 0.01, 0.99))

        # 行业内 z
        def _z(g):
            gnum = g.drop(columns=["_industry_id"])
            mean = gnum.mean(axis=0, skipna=True)
            std = gnum.std(axis=0, ddof=0, skipna=True).replace(0, np.nan)
            z = (gnum - mean) / std
            return z

        z = tmp.groupby("_industry_id", group_keys=False).apply(_z)
        # 若极端情况下全 NaN，退回全市场 z
        if z.isna().all(axis=None):
            gnum = tmp.drop(columns=["_industry_id"])
            mean = gnum.mean(axis=0, skipna=True)
            std = gnum.std(axis=0, ddof=0, skipna=True).replace(0, np.nan)
            z = (gnum - mean) / std

        z["datetime"] = dt
        out.append(z)

    if not out:
        return pd.DataFrame(index=df_vals.index, columns=df_vals.columns, dtype="float32")

    res = pd.concat(out, axis=0).set_index("datetime", append=True).sort_index()
    res = res.reindex(index=df_vals.index, columns=df_vals.columns)
    return res.astype("float32")

def _load_funda_raw() -> Optional[pd.DataFrame]:
    """
    读取基本面源字段 parquet；若不存在返回 None。
    """
    if not CFG.funda_day_file.exists():
        warnings.warn(f"未找到基本面源字段文件：{CFG.funda_day_file}，将仅使用量价特征。")
        return None
    df = pd.read_parquet(CFG.funda_day_file)
    if not isinstance(df.index, pd.MultiIndex):
        try:
            df.index = pd.MultiIndex.from_tuples(df.index, names=["order_book_id", "datetime"])
        except Exception:
            raise ValueError("funda_factors.parquet 需为 MultiIndex(order_book_id, datetime) 索引")
    # 确保 datetime 类型
    dt = df.index.get_level_values("datetime")
    if not np.issubdtype(dt.dtype, np.datetime64):
        df.index = pd.MultiIndex.from_tuples([(sid, pd.to_datetime(x)) for sid, x in df.index],
                                             names=["order_book_id", "datetime"])
    return df.sort_index()

def _align_and_ffill_to_trading_days(df: pd.DataFrame, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    """
    将日频基本面数据对齐至交易日并前向填充，避免采样日缺口。
    """
    sids = df.index.get_level_values(0).unique()
    full_index = pd.MultiIndex.from_product([sids, trading_days], names=["order_book_id", "datetime"])
    df2 = df.reindex(full_index).sort_index()
    df2 = df2.groupby(level=0).ffill()
    return df2

def build_fundamental_cross_section(
    trading_days: pd.DatetimeIndex,
    sample_days: pd.DatetimeIndex
) -> Tuple[pd.DataFrame, List[str]]:
    """
    计算“仅在采样日”的基本面横截面特征（行业内 z 后）与掩码列。
    返回：
    - funda_cs: MultiIndex(order_book_id, datetime=采样日) 的 DataFrame
    - funda_cols: 特征列名（按覆盖率筛选后）
    """
    raw = _load_funda_raw()
    if raw is None or raw.empty:
        return pd.DataFrame(), []

    # 对齐交易日并前向填充
    raw_ff = _align_and_ffill_to_trading_days(raw, trading_days)

    # 构造派生特征（逐日）
    df = raw_ff.copy()
    # 价值/EV
    df["inv_pb"] = 1.0 / (df["pb_ratio_lf"].replace(0, np.nan))
    df["inv_ps"] = 1.0 / (df["ps_ratio_ttm"].replace(0, np.nan))
    df["ev_to_sales"] = df["ev_ttm"] / (df["operating_revenue_ttm_0"].abs() + EPS)
    # 盈利/成长
    df["profit_margin_ttm"] = df["net_profit_parent_company_ttm_0"] / (df["operating_revenue_ttm_0"].abs() + EPS)
    df["revenue_yoy_ttm"] = df.groupby(level=0)["operating_revenue_ttm_0"].pct_change(_TRADING_DAYS_PER_YEAR)
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

    # 仅保留采样日
    df = df[df.index.get_level_values(1).isin(sample_days)]

    # 行业内 z：除掩码外
    df_base = df[_FUNDA_BASE_COLS].copy()

    # 行业映射（utils.load_industry_map：读取中文行业名 -> 稳定整数 ID）
    stock2ind = load_industry_map(CFG.industry_map_file)
    # 将 df_base 的索引股票映射为行业 ID
    # dates 列表
    dates = sorted(df.index.get_level_values(1).unique().tolist())
    df_z = _industry_zscore_per_date(df_base, stock2ind, dates, winsorize_cols=_HEAVY_TAIL_COLS)

    # 覆盖率检查与降级
    keep_cols: List[str] = []
    for col in df_z.columns:
        valid_ratio = 1.0 - df_z[col].isna().mean()
        if valid_ratio < _COVERAGE_THRESHOLD:
            warnings.warn(f"[降级] 基本面特征 {col} 覆盖率 {valid_ratio:.1%} < {_COVERAGE_THRESHOLD:.0%}，跳过该列。")
        else:
            keep_cols.append(col)
    df_z = df_z[keep_cols]

    # 拼接掩码列
    funda_cs = pd.concat([df_z, df[_FUNDA_MASK_COLS]], axis=1)
    funda_cols = list(df_z.columns) + _FUNDA_MASK_COLS

    # 缺失填 0（行业内 z 的均值近 0；掩码为 0/1）
    funda_cs = funda_cs.astype("float32")
    return funda_cs, funda_cols

# ===== 因子计算（与原版一致，量价） =====
def calc_factors_one_stock_full(df: pd.DataFrame) -> pd.DataFrame:
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

# ===== 并行包装 =====
def _compute_one(stock: str, df_all: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
    try:
        hist = df_all.loc[stock]
        if isinstance(hist, pd.Series):
            hist = hist.to_frame().T
        hist = hist.sort_index()
        fct = calc_factors_one_stock_full(hist)
        return stock, fct
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
    # 1) 读取最新的日频数据并规范索引
    day_df = pd.read_parquet(CFG.price_day_file)
    day_df = ensure_multiindex_price(day_df)

    stocks = day_df.index.get_level_values(0).unique().tolist()

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
    print(f"读取局部行情片段: {sliced.index.get_level_values('datetime').min().date()} ~ {sliced.index.get_level_values('datetime').max().date()}, 股票数={sliced.index.get_level_values(0).nunique()}")

    # 5) 并行计算“局部片段”的全量因子缓存（量价）
    stock_feat: Dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=CFG.num_workers) as ex:
        futures = {ex.submit(_compute_one, s, sliced): s for s in stocks if s in sliced.index.get_level_values(0)}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="并行计算因子(局部片段)"):
            s = futures[fut]
            stk, fct = fut.result()
            if not fct.empty:
                stock_feat[stk] = fct

    if len(stock_feat) == 0:
        print("局部片段内无可计算的股票因子，增量构建结束。")
        return

    # 6) 因子列名与历史一致（量价）
    factor_cols_price = enforce_factor_cols_consistency(stock_feat, existed_cols)

    # 6.1 基本面横截面特征（仅采样日）—— 新增
    cal = load_calendar(Path(CFG.trading_day_file))
    # 采样日（周五）
    sample_fridays = pd.DatetimeIndex([d for d in cal if d in missing_fridays])
    # 基本面横截面（行业内 z）
    funda_cs, funda_cols = build_fundamental_cross_section(trading_days=cal, sample_days=sample_fridays)

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
                    if isinstance(gd, bytes):
                        gd = gd.decode('utf-8')
                    existing_groups[pd.Timestamp(gd)] = k
            except Exception:
                continue

        # 更新/写入 factor_cols（量价 + 基本面）
        final_factor_cols = list(factor_cols_price)
        if len(funda_cols) > 0:
            # 将基本面列追加到末尾
            for c in funda_cols:
                if c not in final_factor_cols:
                    final_factor_cols.append(c)
        h5f.attrs['factor_cols'] = np.asarray(final_factor_cols, dtype=str_dt)

        group_idx = next_idx
        for d in tqdm(missing_fridays, desc="增量写入特征仓"):
            # 覆盖：如果已存在该日期组，先删除
            if d in existing_groups:
                try:
                    del h5f[existing_groups[d]]
                except Exception:
                    pass  # 删除失败则忽略，继续用追加写入覆盖（可能留下重复组）

            feats_all = []
            stk_list  = []

            # 对所有股票在该日期切出量价窗口，并在通道维拼接基本面常数序列
            for stk, fct_full in stock_feat.items():
                fct_hist = fct_full[fct_full.index <= d]
                if len(fct_hist) < CFG.daily_window:
                    continue
                fct_win = fct_hist.tail(CFG.daily_window)
                fct_win = fct_win.reindex(columns=factor_cols_price)
                price_vals = fct_win.values  # [T, C_price]
                if not np.isfinite(price_vals).any() or np.isnan(price_vals).all():
                    continue
                price_vals = np.nan_to_num(price_vals, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

                # 基本面：取该采样日横截面值，并复制为长度 T 的常数序列（若无基本面，则仅量价）
                if len(funda_cols) > 0 and (stk, d) in funda_cs.index:
                    row = funda_cs.loc[(stk, d), funda_cols].to_numpy(dtype=np.float32)
                    row = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
                    funda_vals = np.repeat(row[None, :], CFG.daily_window, axis=0)  # [T, C_funda]
                    vals = np.concatenate([price_vals, funda_vals], axis=1)        # [T, C_price+C_funda]
                else:
                    vals = price_vals

                if vals.ndim != 2 or vals.shape[0] != CFG.daily_window:
                    continue

                feats_all.append(vals.astype(np.float32))
                stk_list.append(stk)

            if not feats_all:
                continue

            feats_arr = np.stack(feats_all, axis=0)  # [N,T,C_total]
            # 对整体做一次 MAD 裁剪（与你原流程一致）
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