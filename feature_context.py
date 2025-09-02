# coding: utf-8
"""
生成市场/风格上下文特征：context_features.parquet
- 输入：
  * CFG.index_day_file: 包含 000300.XSHG / 000905.XSHG / 000852.XSHG，MultiIndex(order_book_id, date)
  * CFG.style_day_file: 包含 ['周期风格','成长风格','消费风格','稳定风格','金融风格']，列 sector + 日频行情，index 为整数或日期列
- 输出：
  * data/processed/context_features.parquet（index=周五日期），每行一个上下文向量
- 注意：严格校验，发现异常直接报错（不做静默修复）
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
    # 校验索引必须是 MultiIndex(order_book_id, date)
    if not isinstance(index_df.index, pd.MultiIndex) or index_df.index.nlevels != 2:
        raise ValueError(f"index_day_file 需要 MultiIndex(order_book_id, date)，当前 index={index_df.index}")
    if index_df.index.names != ['order_book_id', 'date']:
        # 放宽：只要含这两个层级即可
        expected = set(['order_book_id', 'date'])
        if set(index_df.index.names) != expected:
            raise ValueError(f"index_day_file 索引级别需包含 ['order_book_id','date']，当前 {index_df.index.names}")

    if 'close' not in index_df.columns or 'total_turnover' not in index_df.columns:
        raise ValueError("index_day_file 需要包含列 ['close','total_turnover']")

    index_df = index_df.sort_index()
    close_pivot = index_df['close'].unstack(0)
    turn_pivot  = index_df['total_turnover'].unstack(0)

    # 严格校验：若缺少目标指数列，直接报错（不做自动补空列）
    missing_idx = [c for c in INDEXS if c not in close_pivot.columns]
    if missing_idx:
        raise ValueError(f"指数数据缺少列: {missing_idx}，请确保 index_day_file 包含这些指数的行情")
    close_pivot = close_pivot.reindex(columns=INDEXS)
    turn_pivot  = turn_pivot.reindex(columns=INDEXS)
    return close_pivot, turn_pivot

def _prepare_sector(sector_df: pd.DataFrame):
    # 要求必须存在 date 列（你的数据描述明确有 date 字段）
    if 'date' not in sector_df.columns:
        # 若索引是 DatetimeIndex，可接收，但仍建议有 date 列
        if isinstance(sector_df.index, pd.DatetimeIndex):
            df = sector_df.reset_index().rename(columns={'index': 'date'})
        else:
            raise ValueError("style_day_file 必须包含 'date' 列或将日期作为 DatetimeIndex")
    else:
        df = sector_df.copy()

    # 必须存在 sector 列
    if 'sector' not in df.columns:
        raise ValueError("style_day_file 需要包含 'sector' 列（中文风格名）")

    req_cols = ['close', 'total_turnover']
    miss = [c for c in req_cols if c not in df.columns]
    if miss:
        raise ValueError(f"style_day_file 缺少必要行情列: {miss}")

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # 透视
    close_pivot = df.pivot(index='date', columns='sector', values='close')
    turn_pivot  = df.pivot(index='date', columns='sector', values='total_turnover')

    # 严格校验：缺少目标风格就报错（不做自动补空列）
    missing_sec = [s for s in SECTORS if s not in close_pivot.columns]
    if missing_sec:
        raise ValueError(f"风格数据缺少列: {missing_sec}，请确保 style_day_file 中包含这些风格的行情")

    close_pivot = close_pivot.reindex(columns=SECTORS)
    turn_pivot  = turn_pivot.reindex(columns=SECTORS)
    return close_pivot, turn_pivot

def build_context(index_df: pd.DataFrame, sector_df: pd.DataFrame,
                  fridays: pd.DatetimeIndex) -> pd.DataFrame:
    idx_close, idx_turn = _prepare_index(index_df)
    sec_close, sec_turn = _prepare_sector(sector_df)

    idx_ret = idx_close.pct_change()
    sec_ret = sec_close.pct_change()

    # 风格 breadth：上涨风格数量 - 下跌风格数量，再除以总风格数
    sec_up = (sec_ret > 0).sum(axis=1)
    sec_dn = (sec_ret < 0).sum(axis=1)
    bread  = (sec_up - sec_dn) / sec_ret.shape[1]

    rows = []
    dates = []

    for d in fridays:
        # 对齐指数：找 <= d 的最后交易日
        idx_dates = idx_close.index[idx_close.index <= d]
        sec_dates = sec_close.index[sec_close.index <= d]
        if len(idx_dates) == 0 or len(sec_dates) == 0:
            # 没有任何可用数据，直接报错，便于定位
            raise ValueError(f"在对齐周五 {d.date()} 时，指数或风格数据无可用交易日（<=该周五），请检查时间范围与数据覆盖。")
        dt_idx = idx_dates[-1]
        dt_sec = sec_dates[-1]

        feat = {}

        # 指数特征
        for code in INDEXS:
            r = idx_ret[code]; t = idx_turn[code]
            # 校验对齐日存在
            if dt_idx not in r.index or dt_idx not in t.index:
                raise ValueError(f"指数 {code} 在对齐日 {dt_idx.date()} 无数据")
            feat[f'{code}_ret_1d'] = r.loc[dt_idx]
            for n in [5, 10, 20, 60]:
                r_mean = _mean(r, n); r_std = _std(r, n)
                t_mean = _mean(t, n); t_std = _std(t, n)
                for ser, name in [(r_mean, 'ret_mean'), (r_std, 'ret_std'),
                                  (t_mean, 'to_mean_ratio'), (t_std, 'to_std_ratio')]:
                    if dt_idx not in ser.index:
                        raise ValueError(f"指数 {code} 在对齐日 {dt_idx.date()} 缺少 {name}_{n}")
                feat[f'{code}_ret_mean_{n}'] = r_mean.loc[dt_idx]
                feat[f'{code}_ret_std_{n}']  = r_std.loc[dt_idx]
                feat[f'{code}_to_mean_ratio_{n}'] = (t_mean / (t + EPS)).loc[dt_idx]
                feat[f'{code}_to_std_ratio_{n}']  = (t_std / (t.abs() + EPS)).loc[dt_idx]

        # 风格特征 + breadth
        if dt_sec not in bread.index:
            raise ValueError(f"bread 指标在对齐日 {dt_sec.date()} 缺失")
        feat['bread_style'] = bread.loc[dt_sec]

        for s in SECTORS:
            r = sec_ret[s]; t = sec_turn[s]
            if dt_sec not in r.index or dt_sec not in t.index:
                raise ValueError(f"风格 {s} 在对齐日 {dt_sec.date()} 无数据")
            feat[f'{s}_ret_1d'] = r.loc[dt_sec]
            for n in [5, 10, 20, 60]:
                r_mean = _mean(r, n); r_std = _std(r, n)
                t_mean = _mean(t, n); t_std = _std(t, n)
                for ser, name in [(r_mean, 'ret_mean'), (r_std, 'ret_std'),
                                  (t_mean, 'to_mean_ratio'), (t_std, 'to_std_ratio')]:
                    if dt_sec not in ser.index:
                        raise ValueError(f"风格 {s} 在对齐日 {dt_sec.date()} 缺少 {name}_{n}")
                feat[f'{s}_ret_mean_{n}'] = r_mean.loc[dt_sec]
                feat[f'{s}_ret_std_{n}']  = r_std.loc[dt_sec]
                feat[f'{s}_to_mean_ratio_{n}'] = (t_mean / (t + EPS)).loc[dt_sec]
                feat[f'{s}_to_std_ratio_{n}']  = (t_std / (t.abs() + EPS)).loc[dt_sec]

        rows.append(feat)
        dates.append(d)

    if len(rows) == 0:
        raise ValueError("未能构建任何上下文特征行，请检查输入数据覆盖与时间范围。")

    ctx_df = pd.DataFrame(rows, index=pd.DatetimeIndex(dates, name='date')).sort_index()

    # 严格类型/形状校验
    if not isinstance(ctx_df.index, pd.DatetimeIndex):
        raise AssertionError("context_features 的索引必须是 DatetimeIndex（周五日期）")
    if ctx_df.isna().all(axis=None):
        raise ValueError("context_features 全为 NaN，请检查指数/风格数据或对齐逻辑。")

    # 替换 inf 后仍保留 NaN 以便你发现异常（不做填充）
    ctx_df = ctx_df.replace([np.inf, -np.inf], np.nan)

    return ctx_df

def main():
    print("加载指数与风格数据 ...")
    index_df  = pd.read_parquet(CFG.index_day_file)   # MultiIndex(order_book_id, date)
    sector_df = pd.read_parquet(CFG.style_day_file)   # 列含 sector, date, open... / 或 DatetimeIndex

    cal = load_calendar(CFG.trading_day_file)
    fridays = weekly_fridays(cal)
    fridays = fridays[(fridays >= pd.Timestamp(CFG.start_date)) & (fridays <= pd.Timestamp(CFG.end_date))]

    ctx_df = build_context(index_df, sector_df, fridays)

    out_path = CFG.processed_dir / "context_features.parquet"
    # 若仍含 NaN，你可能希望先看到，再决定是否填充；这里选择直接保存并提示
    if ctx_df.isna().any(axis=None):
        missing_ratio = float(ctx_df.isna().sum().sum()) / (ctx_df.shape[0] * ctx_df.shape[1])
        print(f"警告：context_features 含 NaN，占比 {missing_ratio:.2%}。训练前可自行决定是否填充。")

    ctx_df.to_parquet(out_path)
    print(f"上下文特征生成完成：{out_path}")
    print(f"ctx_dim = {ctx_df.shape[1]} 列")

if __name__ == "__main__":
    main()