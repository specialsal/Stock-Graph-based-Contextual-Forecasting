# coding: utf-8
"""
标签生成模块（增量版） - 计算下周收益率
- 对缺失周五增量生成；并从“最后一个已存在周五”起重算该周五覆盖修订

本版本修改：标签口径由“周五收盘->下周五收盘（C2C）”
调整为“周五下一交易日开盘->下周五收盘（O2C）”。
"""
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from config import CFG
from utils import load_calendar, weekly_fridays

def _align_to_last_available(dates_index: pd.DatetimeIndex, target: pd.Timestamp):
    mask = dates_index <= target
    if not mask.any():
        return None
    return dates_index[mask][-1]

def generate_weekly_labels_incremental(
    daily_df: pd.DataFrame,
    sample_dates: pd.DatetimeIndex,
    save_path: Path
) -> pd.DataFrame:
    """
    增量生成下周收益率标签（O2C 口径）：
    起点：信号周五后的“下一交易日”的开盘价
    终点：下一个周五（或不晚于它的最后一个交易日）的收盘价

    - 仅处理 sample_dates 中尚未存在于 save_path 的周五；
    - 额外包含“最后一个已存在周五”以覆盖修订；
    返回：新增/覆盖的标签 DataFrame（若无新增，返回空）
    """
    # 已有标签
    if save_path.exists():
        label_old = pd.read_parquet(save_path)
        if not isinstance(label_old.index, pd.MultiIndex):
            raise ValueError("已有标签文件的索引应为 MultiIndex(date, stock)")
        existing_dates = pd.DatetimeIndex(sorted(label_old.index.get_level_values('date').unique()))
        last_existing = existing_dates.max() if len(existing_dates) > 0 else None
        print(f"[Label增量] 已有标签组（周五）数量={len(existing_dates)}")
    else:
        label_old = None
        existing_dates = pd.DatetimeIndex([])
        last_existing = None

    # 仅保留“缺失周五且不是最后一个周五”（因为需要 next Friday）
    if len(sample_dates) < 2:
        print("[Label增量] 采样日期不足，跳过。")
        return pd.DataFrame()

    sample_dates = sample_dates.sort_values()
    # 缺失周五（排除最后一个周五）
    missing = [d for d in sample_dates[:-1] if d not in set(existing_dates)]
    # 额外加入“最后一个已存在周五”以覆盖修订（但也要保证它不是最后一个 sample 周五）
    if last_existing is not None and last_existing != sample_dates[-1] and last_existing not in missing:
        missing.append(last_existing)

    missing = pd.DatetimeIndex(sorted(set(missing)))
    if len(missing) == 0:
        print("[Label增量] 无缺失或需覆盖的周五，跳过。")
        return pd.DataFrame()

    print(f"[Label增量] 待新增/覆盖周五数量={len(missing)} (范围: {missing.min().date()} ~ {missing.max().date()})")

    # daily_df 预处理
    if not isinstance(daily_df.index, pd.MultiIndex) or daily_df.index.nlevels != 2:
        raise ValueError("daily_df 必须是 MultiIndex(order_book_id, date)")

    daily_df = daily_df.sort_index()
    if daily_df.index.names != ['order_book_id','date']:
        try:
            daily_df.index = daily_df.index.set_names(['order_book_id','date'])
        except Exception:
            pass

    # 构造开盘/收盘 pivot
    if 'close' not in daily_df.columns or 'open' not in daily_df.columns:
        raise ValueError("daily_df 需包含 'open' 与 'close' 列")
    close_pivot = daily_df['close'].unstack(level='order_book_id')
    open_pivot  = daily_df['open'].unstack(level='order_book_id')
    available_dates = close_pivot.index

    # 确定片段边界：从“最早需要计算的周五”的对齐日起，至“最大周五的下一周五”的对齐日
    first_missing = missing.min()
    last_missing  = missing.max()
    # next Friday of last_missing
    future_mask = sample_dates > last_missing
    next_friday_after_last = sample_dates[future_mask][0] if future_mask.any() else None
    if next_friday_after_last is None:
        # 防御（正常不会触发，因为已排除最后一个周五）
        print("[Label增量] 找不到最后周五的下一周五，跳过。")
        return pd.DataFrame()

    start_date = _align_to_last_available(available_dates, first_missing)
    end_date   = _align_to_last_available(available_dates, next_friday_after_last)
    if start_date is None or end_date is None or start_date >= end_date:
        print("[Label增量] 价格数据覆盖不足，无法计算增量标签。")
        return pd.DataFrame()

    # 截取必要片段（同一日期范围）
    mask_slice = (close_pivot.index >= start_date) & (close_pivot.index <= end_date)
    close_pivot_slice = close_pivot.loc[mask_slice]
    open_pivot_slice  = open_pivot.loc[mask_slice]
    available_dates_slice = close_pivot_slice.index

    weekly_returns = []
    # 我们只遍历 missing（含“最后一个已存在周五”）
    for d in tqdm(missing, desc="增量计算周收益率（O2C）"):
        # 找到下一周五
        d_next_idx = sample_dates.get_indexer([d])[0] + 1
        next_d = sample_dates[d_next_idx]

        # 对齐到不晚于各自周五的最后一个交易日
        cur_aligned = _align_to_last_available(available_dates_slice, d)
        nxt_aligned = _align_to_last_available(available_dates_slice, next_d)
        if cur_aligned is None or nxt_aligned is None:
            continue

        # 起点改为：cur_aligned 的“下一交易日”
        pos = available_dates_slice.get_indexer([cur_aligned])[0]
        if pos < 0 or pos + 1 >= len(available_dates_slice):
            continue
        cur_aligned_next = available_dates_slice[pos + 1]

        # 若下一交易日已超过终点，则跳过
        if cur_aligned_next > nxt_aligned:
            continue

        try:
            start_open = open_pivot_slice.loc[cur_aligned_next]
            end_close  = close_pivot_slice.loc[nxt_aligned]
        except KeyError:
            continue

        returns = (end_close / start_open - 1).replace([np.inf, -np.inf], np.nan)
        valid = (~returns.isna()) & (~start_open.isna()) & (~end_close.isna())
        if valid.any():
            sub = pd.DataFrame({
                'date': d,
                'stock': returns.index[valid],
                'next_week_return': returns[valid].values
            })
            weekly_returns.append(sub)

    if len(weekly_returns) == 0:
        print("[Label增量] 本次未生成新增/覆盖标签。")
        return pd.DataFrame()

    label_new = pd.concat(weekly_returns, ignore_index=True)
    label_new = label_new.set_index(['date','stock']).sort_index()

    # 合并保存（新覆盖旧）
    if label_old is None or label_old.empty:
        label_all = label_new
    else:
        label_all = pd.concat([label_old, label_new], axis=0)
        label_all = label_all[~label_all.index.duplicated(keep='last')].sort_index()

    label_all.to_parquet(save_path)
    print(f"[Label增量] 保存完成：{save_path}，总行数={len(label_all)}，新增/覆盖={len(label_new)}")
    return label_new

def main():
    print("加载日频价格数据...")
    daily_df = pd.read_parquet(CFG.price_day_file)

    calendar = load_calendar(CFG.trading_day_file)
    sample_dates = weekly_fridays(calendar)
    sample_dates = sample_dates[(sample_dates >= pd.Timestamp(CFG.start_date)) &
                                (sample_dates <= pd.Timestamp(CFG.end_date))]

    save_path = CFG.label_file
    _ = generate_weekly_labels_incremental(daily_df, sample_dates, save_path)

if __name__ == "__main__":
    main()