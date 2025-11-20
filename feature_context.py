# coding: utf-8
"""
增量生成市场/风格上下文特征：context_features.parquet
- 对缺失的周五增量计算并追加保存
- 从“最后一个已存在周五”起重新计算该周五（以覆盖修订），再合并去重
- 严格校验输入数据；对齐逻辑与原版一致
"""
import numpy as np
import pandas as pd
from pathlib import Path
from config import CFG
from utils import load_calendar, weekly_fridays

EPS = 1e-12

SECTORS = ['周期风格', '成长风格', '消费风格', '稳定风格', '金融风格']
SECTOR_NAME_MAP = {
    '周期风格': 'cyclical',
    '成长风格': 'growth',
    '消费风格': 'consumption',
    '稳定风格': 'stable',
    '金融风格': 'financial',
}
INDEXS  = ['000300.XSHG', '000905.XSHG', '000852.XSHG']

def load_sector_breadth():
    """从 sector_price_day.parquet 读取每日风格 breadth 并转成宽表"""
    path = CFG.raw_dir / "sector_price_day.parquet"
    if not path.exists():
        raise FileNotFoundError(f"缺少 {path}，无法计算风格 breadth")
    df = pd.read_parquet(path)
    if 'date' not in df.columns or 'sector' not in df.columns or 'breadth' not in df.columns:
        raise ValueError("sector_price_day 需要包含 ['date','sector','breadth'] 列")
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['sector'].isin(SECTORS)].copy()
    br_pivot = df.pivot(index='date', columns='sector', values='breadth')
    # 统一列顺序
    br_pivot = br_pivot.reindex(columns=SECTORS)
    return br_pivot

def load_index_breadth() -> pd.DataFrame:
    """
    使用当前指数成分 + stock_price_day 计算指数成分 breadth
    返回：index=date, columns=INDEXS
    """
    # 个股日行情
    stock_price_path = CFG.price_day_file  # 你在 CFG 里对应的路径名，请用实际字段
    stock_df = pd.read_parquet(stock_price_path)
    if not isinstance(stock_df.index, pd.MultiIndex):
        idx_cols = [c for c in stock_df.columns if c in ['order_book_id', 'datetime']]
        if set(idx_cols) == {'order_book_id', 'datetime'}:
            stock_df['datetime'] = pd.to_datetime(stock_df['datetime'])
            stock_df = stock_df.set_index(['order_book_id', 'datetime']).sort_index()
        else:
            raise ValueError("stock_price_day 必须是 MultiIndex(order_book_id, datetime)")
    stock_df = stock_df[['open', 'close']].copy()
    stock_df = stock_df.reset_index().rename(columns={'datetime': 'date'})
    stock_df['date'] = pd.to_datetime(stock_df['date'])

    # 读取指数成分快照
    comp_dir = CFG.raw_dir
    comp_map = {
        '000300.XSHG': comp_dir / 'index_components_000300.csv',
        '000905.XSHG': comp_dir / 'index_components_000905.csv',
        '000852.XSHG': comp_dir / 'index_components_000852.csv',
    }
    index_breadth_rows = []

    for idx_code, path in comp_map.items():
        if not path.exists():
            raise FileNotFoundError(f"缺少指数成分文件: {path}")
        comps = pd.read_csv(path, index_col=0, header=None).iloc[:,0].astype(str).str.strip().tolist()
        if len(comps) == 0:
            continue

        # 过滤出成分股行情
        sub = stock_df[stock_df['order_book_id'].isin(comps)].copy()
        if sub.empty:
            continue

        def _agg_index_one_day(df_day: pd.DataFrame):
            # 使用 close/open 近似当日收益
            df_day = df_day.dropna(subset=['open','close'])
            if df_day.empty:
                return np.nan
            ret = df_day['close'] / df_day['open'] - 1.0
            up = (ret > 0).sum()
            down = (ret < 0).sum()
            total = len(ret)
            if total == 0:
                return np.nan
            return float((up - down) / total)

        br = sub.groupby('date')[['open', 'close']].apply(_agg_index_one_day)
        br.name = idx_code
        index_breadth_rows.append(br)

    if not index_breadth_rows:
        return pd.DataFrame(index=pd.DatetimeIndex([], name='date'), columns=INDEXS)

    br_df = pd.concat(index_breadth_rows, axis=1)
    br_df.index = pd.to_datetime(br_df.index)
    br_df = br_df.sort_index()
    # 统一列顺序
    br_df = br_df.reindex(columns=INDEXS)
    return br_df

def _ret(s: pd.Series): return s.pct_change()
def _mean(s: pd.Series, n: int): return s.rolling(n, min_periods=1).mean()
def _std (s: pd.Series, n: int): return s.rolling(n, min_periods=1).std()

# feature_context.py
def _prepare_index(index_df: pd.DataFrame):
    if not isinstance(index_df.index, pd.MultiIndex) or index_df.index.nlevels != 2:
        raise ValueError(f"index_day_file 需要 MultiIndex(order_book_id, date/datetime)，当前 index={index_df.index}")

    names = list(index_df.index.names)
    # 兼容 ('order_book_id','datetime')
    if set(names) == set(['order_book_id', 'datetime']):
        index_df = index_df.copy()
        index_df.index = index_df.index.set_names(['order_book_id', 'date'])
    elif set(names) != set(['order_book_id', 'date']):
        raise ValueError(f"index_day_file 索引级别需包含 ['order_book_id','date'] (或 'datetime')，当前 {names}")

    if 'close' not in index_df.columns or 'total_turnover' not in index_df.columns:
        raise ValueError("index_day_file 需要包含列 ['close','total_turnover']")

    index_df = index_df.sort_index()
    close_pivot = index_df['close'].unstack(0)
    turn_pivot  = index_df['total_turnover'].unstack(0)

    missing_idx = [c for c in INDEXS if c not in close_pivot.columns]
    if missing_idx:
        raise ValueError(f"指数数据缺少列: {missing_idx}，请确保 index_day_file 覆盖这些指数")

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
                            sector_breadth: pd.DataFrame,
                            index_breadth: pd.DataFrame,
                            target_fridays: pd.DatetimeIndex) -> pd.DataFrame:
    """仅对给定的 target_fridays 计算上下文特征
       sector_breadth: index=date, columns=SECTORS, 值为成分股聚合的 breadth
    """
    idx_close, idx_turn = _prepare_index(index_df)
    sec_close, sec_turn = _prepare_sector(sector_df)

    idx_ret = idx_close.pct_change()
    sec_ret = sec_close.pct_change()

    rows = []
    dates = []

    for d in target_fridays:
        idx_dates = idx_close.index[idx_close.index <= d]
        sec_dates = sec_close.index[sec_close.index <= d]
        if len(idx_dates) == 0 or len(sec_dates) == 0:
            continue
        dt_idx = idx_dates[-1]
        dt_sec = sec_dates[-1]

        feat = {}

        # 1) 指数特征（原逻辑不变）
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
        # 1.5) 宽基指数 breadth（基于成分股）
        if index_breadth is not None and dt_idx in index_breadth.index:
            for code in INDEXS:
                if code in index_breadth.columns:
                    feat[f'breadth_{code[3:-5]}'] = index_breadth.at[dt_idx, code]
        # 2) 风格 breadth：用 sector_breadth（5 个因子）
        if sector_breadth is not None and dt_sec in sector_breadth.index:
            for s in SECTORS:
                val = sector_breadth.at[dt_sec, s]
                eng = SECTOR_NAME_MAP.get(s, s)  # 找不到就退回原名
                feat[f'breadth_{eng}'] = val
            # 若仍需要一个总的风格 breadth，可用简单平均
            vals = sector_breadth.loc[dt_sec, SECTORS]
            feat['bread_style'] = vals.mean()

        # 3) 风格收益与成交额特征（原逻辑不变）
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

        # 宽基指数 breadth 会在下一个小节添加

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
        existing_dates = pd.DatetimeIndex(sorted(set(ctx_old.index)))
        last_existing = existing_dates.max() if len(existing_dates) > 0 else None
        print(f"[Context增量] 已有行数={len(ctx_old)}, 覆盖周五={len(existing_dates)}")
    else:
        ctx_old = None
        existing_dates = pd.DatetimeIndex([])
        last_existing = None

    # 需要增量的周五（包含：缺失周五 ∪ 最后一个已存在周五，用于覆盖修订）
    missing = [d for d in fridays_all if d not in set(existing_dates)]
    if last_existing is not None and last_existing not in missing:
        missing.append(last_existing)
    missing = pd.DatetimeIndex(sorted(set(missing)))

    if len(missing) == 0:
        print("[Context增量] 无需更新，跳过。")
        return

    print(f"[Context增量] 待计算周五数量={len(missing)} (范围: {missing.min().date()} ~ {missing.max().date()})")

    # 载入原始数据
    index_df  = pd.read_parquet(CFG.index_day_file)
    sector_df = pd.read_parquet(CFG.style_day_file)
    sector_breadth = load_sector_breadth()
    index_breadth = load_index_breadth()

    # 仅对目标周五计算
    ctx_new = build_context_for_dates(index_df, sector_df, sector_breadth, index_breadth, missing)

    if ctx_old is None or ctx_old.empty:
        ctx_all = ctx_new
    else:
        ctx_all = pd.concat([ctx_old, ctx_new], axis=0)
        # 新结果覆盖旧（从“最后一日”起重算，必须 keep='last'）
        ctx_all = ctx_all[~ctx_all.index.duplicated(keep='last')].sort_index()

    ctx_all.to_parquet(out_path)
    print(f"[Context增量] 保存完成：{out_path}，总行数={len(ctx_all)}，新增或覆盖={len(ctx_new)}")

if __name__ == "__main__":
    main()