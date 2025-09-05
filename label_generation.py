# coding: utf-8
"""
标签生成模块（增量版） - 计算下周收益率
- 仅对缺失周五生成并追加
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
    增量生成下周收益率标签，仅处理 sample_dates 中尚未存在于 save_path 的周五。
    返回：新增的标签 DataFrame（若无新增，返回空）
    """
    # 已有标签
    if save_path.exists():
        label_old = pd.read_parquet(save_path)
        if not isinstance(label_old.index, pd.MultiIndex):
            raise ValueError("已有标签文件的索引应为 MultiIndex(date, stock)")
        existing_dates = set(label_old.index.get_level_values('date').unique())
        print(f"[Label增量] 已有标签组（周五）数量={len(existing_dates)}")
    else:
        label_old = None
        existing_dates = set()

    # 仅保留“缺失周五且不是最后一个周五”（因为需要 next Friday）
    if len(sample_dates) < 2:
        print("[Label增量] 采样日期不足，跳过。")
        return pd.DataFrame()

    sample_dates = sample_dates.sort_values()
    missing = [d for d in sample_dates[:-1] if d not in existing_dates]
    missing = pd.DatetimeIndex(sorted(missing))
    if len(missing) == 0:
        print("[Label增量] 无缺失周五，跳过。")
        return pd.DataFrame()

    print(f"[Label增量] 待新增周五数量={len(missing)} (范围: {missing.min().date()} ~ {missing.max().date()})")

    # pivot close
    if not isinstance(daily_df.index, pd.MultiIndex) or daily_df.index.nlevels != 2:
        raise ValueError("daily_df 必须是 MultiIndex(order_book_id, date)")

    daily_df = daily_df.sort_index()
    if daily_df.index.names != ['order_book_id','date']:
        try:
            daily_df.index = daily_df.index.set_names(['order_book_id','date'])
        except Exception:
            pass

    close_pivot = daily_df['close'].unstack(level='order_book_id')
    available_dates = close_pivot.index

    # 确定片段边界：从最早缺失周五的对齐日起，至“最大缺失周五的下一周五”的对齐日
    first_missing = missing.min()
    last_missing  = missing.max()
    # next Friday of last_missing
    future_mask = sample_dates > last_missing
    next_friday_after_last = sample_dates[future_mask][0] if future_mask.any() else None
    if next_friday_after_last is None:
        # 理论上不会发生，因为我们已排除最后一个周五，但加一道保护
        print("[Label增量] 找不到最后缺失周五的下一周五，跳过。")
        return pd.DataFrame()

    start_date = _align_to_last_available(available_dates, first_missing)
    end_date   = _align_to_last_available(available_dates, next_friday_after_last)
    if start_date is None or end_date is None or start_date >= end_date:
        print("[Label增量] 价格数据覆盖不足，无法计算增量标签。")
        return pd.DataFrame()

    # 截取必要片段
    close_pivot_slice = close_pivot.loc[(close_pivot.index >= start_date) & (close_pivot.index <= end_date)]
    available_dates_slice = close_pivot_slice.index

    weekly_returns = []
    # 我们只遍历“缺失周五”
    for d in tqdm(missing, desc="增量计算周收益率"):
        d_next_idx = sample_dates.get_indexer([d])[0] + 1
        next_d = sample_dates[d_next_idx]

        cur_aligned = _align_to_last_available(available_dates_slice, d)
        nxt_aligned = _align_to_last_available(available_dates_slice, next_d)
        if cur_aligned is None or nxt_aligned is None or cur_aligned >= nxt_aligned:
            continue

        try:
            start_price = close_pivot_slice.loc[cur_aligned]
            end_price   = close_pivot_slice.loc[nxt_aligned]
        except KeyError:
            continue

        returns = (end_price / start_price - 1).replace([np.inf, -np.inf], np.nan)
        valid = (~returns.isna()) & (~start_price.isna())
        if valid.any():
            sub = pd.DataFrame({
                'date': d,
                'stock': returns.index[valid],
                'next_week_return': returns[valid].values
            })
            weekly_returns.append(sub)

    if len(weekly_returns) == 0:
        print("[Label增量] 本次未生成新增标签。")
        return pd.DataFrame()

    label_new = pd.concat(weekly_returns, ignore_index=True)
    label_new = label_new.set_index(['date','stock']).sort_index()

    # 合并保存
    if label_old is None:
        label_all = label_new
    else:
        label_all = pd.concat([label_old, label_new], axis=0)
        label_all = label_all[~label_all.index.duplicated(keep='first')].sort_index()

    label_all.to_parquet(save_path)
    print(f"[Label增量] 保存完成：{save_path}，总行数={len(label_all)}，新增={len(label_new)}")
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