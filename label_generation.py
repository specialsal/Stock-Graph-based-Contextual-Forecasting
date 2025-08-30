# coding: utf-8
"""
标签生成模块 - 计算下周收益率
"""
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from config import CFG
from utils import load_trading_calendar, week_last_trading_days


def generate_weekly_labels(
        daily_df: pd.DataFrame,
        sample_dates: pd.DatetimeIndex,
        calendar: pd.DatetimeIndex,
        save_path: Path = None
) -> pd.DataFrame:
    """
    生成下周收益率标签

    Parameters:
    -----------
    daily_df: 日频价格数据，MultiIndex (order_book_id, date)
    sample_dates: 采样日期（每周五）
    calendar: 交易日历
    save_path: 保存路径

    Returns:
    --------
    label_df: MultiIndex DataFrame (date, stock) -> next_week_return
    """
    print("开始生成周度收益率标签...")

    # 将close价格转换为pivot表格式
    close_pivot = daily_df['close'].unstack(level='order_book_id')

    weekly_returns = []

    for i in tqdm(range(len(sample_dates) - 1), desc="计算周收益率"):
        current_friday = sample_dates[i]
        next_friday = sample_dates[i + 1]

        # 找到下周的交易日
        next_week_mask = (calendar > current_friday) & (calendar <= next_friday)
        next_week_days = calendar[next_week_mask]

        if len(next_week_days) == 0:
            continue

        # 确保日期在数据中
        if current_friday not in close_pivot.index:
            # 如果当前周五不是交易日，找最近的交易日
            idx = calendar.get_indexer([current_friday], method='ffill')[0]
            if idx < 0:
                continue
            current_friday = calendar[idx]

        if next_week_days[-1] not in close_pivot.index:
            continue

        # 计算收益率
        start_price = close_pivot.loc[current_friday]
        end_price = close_pivot.loc[next_week_days[-1]]

        # 计算收益率，处理缺失值
        returns = (end_price / start_price - 1).replace([np.inf, -np.inf], np.nan)

        # 构建标签数据
        for stock in returns.index:
            if not pd.isna(returns[stock]):
                weekly_returns.append({
                    'date': sample_dates[i],  # 使用原始的周五日期
                    'stock': stock,
                    'next_week_return': returns[stock]
                })

    # 转换为DataFrame
    label_df = pd.DataFrame(weekly_returns)

    if len(label_df) > 0:
        label_df = label_df.set_index(['date', 'stock'])

        # 保存标签
        if save_path:
            label_df.to_parquet(save_path)
            print(f"标签已保存至: {save_path}")
            print(f"标签数量: {len(label_df)}")
            print(f"标签统计:\n{label_df['next_week_return'].describe()}")
    else:
        raise ValueError("未生成任何标签数据")

    return label_df


def main():
    """主函数 - 生成并保存标签"""
    # 加载数据
    print("加载日频价格数据...")
    daily_df = pd.read_parquet(CFG.price_daily_file)

    # 加载交易日历
    calendar = load_trading_calendar(CFG.trading_day_file)

    # 获取每周最后交易日
    sample_dates = week_last_trading_days(calendar)

    # 生成并保存标签
    label_df = generate_weekly_labels(
        daily_df,
        sample_dates,
        calendar,
        save_path=CFG.label_file
    )

    return label_df


if __name__ == "__main__":
    main()