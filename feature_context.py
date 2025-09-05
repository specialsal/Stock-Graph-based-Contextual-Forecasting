# coding: utf-8
"""
增量生成市场/风格上下文特征：context_features.parquet
- 仅对缺失的周五增量计算并追加保存
- 严格校验输入数据；对齐逻辑与原版一致
"""
import numpy as np
import pandas as pd
from pathlib import Path
from config import CFG
from utils import load_calendar, weekly_fridays

EPS = 1e-12

SECTORS = ['周期风格', '成长风格', '消费风格', '稳定风格', '金融风格']
INDEXS  = ['000300.XSHG', '000905.XSHG', '000852.XSHG']

def _ret(s: pd.Series): return s.pct_change()
def _mean(s: pd.Series, n: int): return s.rolling(n, min_periods=1).mean()
def _std (s: pd.Series, n: int): return s.rolling(n, min_periods=1).std()

def _prepare_index(index_df: pd.DataFrame):
    if not isinstance(index_df.index, pd.MultiIndex) or index_df.index.nlevels != 2:
        raise ValueError(f"index_day_file 需要 MultiIndex(order_book_id, date)，当前 index={index_df.index}")
    if set(index_df.index.names) != set(['order_book_id','date']):
        raise ValueError(f"index_day_file 索引级别需包含 ['order_book_id','date']，当前 {index_df.index.names}")

    if 'close' not in index_df.columns or 'total_turnover' not in index_df.columns:
        raise ValueError("index_day_file 需要包含列 ['close','total_turnover']")

    index_df = index_df.sort_index()
    close_pivot = index_df['close'].unstack(0)
    turn_pivot  = index_df['total_turnover'].unstack(0)

    missing_idx = [c for c in INDEXS if c not in close_pivot.columns]
    if missing_idx:
        raise ValueError(f"指数数据缺少列: {missing_idx}，请确保 index_day_file 包含这些指数的行情")
    close_pivot = close_pivot.reindex(columns=INDEXS)
    turn_pivot  = turn_pivot.reindex(columns=INDEXS)
    return close_pivot, turn_pivot

def _prepare_sector(sector_df: pd.DataFrame):
    if 'date' not in sector_df.columns:
        if isinstance(sector_df.index, pd.DatetimeIndex):
            df = sector_df.reset_index().rename(columns={'index': 'date'})
        else:
            raise ValueError("style_day_file 必须包含 'date' 列或将日期作为 DatetimeIndex")
    else:
        df = sector_df.copy()

    if 'sector' not in df.columns:
        raise ValueError("style_day_file 需要包含 'sector' 列（中文风格名）")

    req_cols = ['close', 'total_turnover']
    miss = [c for c in req_cols if c not in df.columns]
    if miss:
        raise ValueError(f"style_day_file 缺少必要行情列: {miss}")

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    close_pivot = df.pivot(index='date', columns='sector', values='close')
    turn_pivot  = df.pivot(index='date', columns='sector', values='total_turnover')

    missing_sec = [s for s in SECTORS if s not in close_pivot.columns]
    if missing_sec:
        raise ValueError(f"风格数据缺少列: {missing_sec}，请确保 style_day_file 中包含这些风格的行情")

    close_pivot = close_pivot.reindex(columns=SECTORS)
    turn_pivot  = turn_pivot.reindex(columns=SECTORS)
    return close_pivot, turn_pivot

def build_context_for_dates(index_df: pd.DataFrame,
                            sector_df: pd.DataFrame,
                            target_fridays: pd.DatetimeIndex) -> pd.DataFrame:
    """仅对给定的 target_fridays 计算上下文特征"""
    idx_close, idx_turn = _prepare_index(index_df)
    sec_close, sec_turn = _prepare_sector(sector_df)

    idx_ret = idx_close.pct_change()
    sec_ret = sec_close.pct_change()

    sec_up = (sec_ret > 0).sum(axis=1)
    sec_dn = (sec_ret < 0).sum(axis=1)
    bread  = (sec_up - sec_dn) / sec_ret.shape[1]

    rows = []
    dates = []

    for d in target_fridays:
        idx_dates = idx_close.index[idx_close.index <= d]
        sec_dates = sec_close.index[sec_close.index <= d]
        if len(idx_dates) == 0 or len(sec_dates) == 0:
            # 没有可用数据，直接跳过该周五（保持严格，但允许数据边界导致的缺失）
            continue
        dt_idx = idx_dates[-1]
        dt_sec = sec_dates[-1]

        feat = {}
        # 指数特征
        for code in INDEXS:
            r = idx_ret[code]; t = idx_turn[code]
            if dt_idx not in r.index or dt_idx not in t.index:
                continue
            feat[f'{code}_ret_1d'] = r.loc[dt_idx]
            for n in [5, 10, 20, 60]:
                r_mean = _mean(r, n); r_std = _std(r, n)
                t_mean = _mean(t, n); t_std = _std(t, n)
                if dt_idx not in r_mean.index or dt_idx not in r_std.index or dt_idx not in t_mean.index or dt_idx not in t_std.index:
                    continue
                feat[f'{code}_ret_mean_{n}'] = r_mean.loc[dt_idx]
                feat[f'{code}_ret_std_{n}']  = r_std.loc[dt_idx]
                feat[f'{code}_to_mean_ratio_{n}'] = (t_mean / (t + EPS)).loc[dt_idx]
                feat[f'{code}_to_std_ratio_{n}']  = (t_std / (t.abs() + EPS)).loc[dt_idx]

        # 风格 breadth
        if dt_sec in bread.index:
            feat['bread_style'] = bread.loc[dt_sec]

        # 风格特征
        for s in SECTORS:
            r = sec_ret[s]; t = sec_turn[s]
            if dt_sec not in r.index or dt_sec not in t.index:
                continue
            feat[f'{s}_ret_1d'] = r.loc[dt_sec]
            for n in [5, 10, 20, 60]:
                r_mean = _mean(r, n); r_std = _std(r, n)
                t_mean = _mean(t, n); t_std = _std(t, n)
                if dt_sec not in r_mean.index or dt_sec not in r_std.index or dt_sec not in t_mean.index or dt_sec not in t_std.index:
                    continue
                feat[f'{s}_ret_mean_{n}'] = r_mean.loc[dt_sec]
                feat[f'{s}_ret_std_{n}']  = r_std.loc[dt_sec]
                feat[f'{s}_to_mean_ratio_{n}'] = (t_mean / (t + EPS)).loc[dt_sec]
                feat[f'{s}_to_std_ratio_{n}']  = (t_std / (t.abs() + EPS)).loc[dt_sec]

        if len(feat) == 0:
            continue
        rows.append(feat)
        dates.append(d)

    if len(rows) == 0:
        return pd.DataFrame(index=pd.DatetimeIndex([], name='date'))

    ctx_df = pd.DataFrame(rows, index=pd.DatetimeIndex(dates, name='date')).sort_index()
    ctx_df = ctx_df.replace([np.inf, -np.inf], np.nan)
    return ctx_df

def main():
    out_path = CFG.processed_dir / "context_features.parquet"

    # 交易周五全集
    cal = load_calendar(CFG.trading_day_file)
    fridays_all = weekly_fridays(cal)
    fridays_all = fridays_all[(fridays_all >= pd.Timestamp(CFG.start_date)) &
                              (fridays_all <= pd.Timestamp(CFG.end_date))]

    # 已有 context 索引
    if out_path.exists():
        ctx_old = pd.read_parquet(out_path)
        if not isinstance(ctx_old.index, pd.DatetimeIndex):
            raise AssertionError("context_features 的索引必须是 DatetimeIndex")
        existing_dates = set(ctx_old.index)
        print(f"[Context增量] 已有行数={len(ctx_old)}, 覆盖周五={len(existing_dates)}")
    else:
        ctx_old = None
        existing_dates = set()

    # 需要增量的周五
    missing = [d for d in fridays_all if d not in existing_dates]
    missing = pd.DatetimeIndex(sorted(missing))
    if len(missing) == 0:
        print("[Context增量] 无缺失周五，跳过。")
        return

    print(f"[Context增量] 待新增周五数量={len(missing)} (范围: {missing.min().date()} ~ {missing.max().date()})")

    # 载入原始数据
    index_df  = pd.read_parquet(CFG.index_day_file)
    sector_df = pd.read_parquet(CFG.style_day_file)

    # 仅对缺失周五计算
    ctx_new = build_context_for_dates(index_df, sector_df, missing)

    if ctx_old is None:
        ctx_all = ctx_new
    else:
        ctx_all = pd.concat([ctx_old, ctx_new], axis=0)
        ctx_all = ctx_all[~ctx_all.index.duplicated(keep='first')].sort_index()

    ctx_all.to_parquet(out_path)
    print(f"[Context增量] 保存完成：{out_path}，总行数={len(ctx_all)}，新增={len(ctx_new)}")

if __name__ == "__main__":
    main()