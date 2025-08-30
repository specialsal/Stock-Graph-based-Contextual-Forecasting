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

    # 获取价格数据中实际存在的日期
    available_dates = close_pivot.index
    print(f"价格数据日期范围: {available_dates.min()} 到 {available_dates.max()}")
    print(f"采样日期范围: {sample_dates.min()} 到 {sample_dates.max()}")

    weekly_returns = []

    for i in tqdm(range(len(sample_dates) - 1), desc="计算周收益率"):
        current_friday = sample_dates[i]
        next_friday = sample_dates[i + 1]

        # 找到在价格数据中实际存在的最近交易日
        # 对于当前周五
        current_dates_mask = available_dates <= current_friday
        if not current_dates_mask.any():
            continue
        actual_current_date = available_dates[current_dates_mask][-1]

        # 对于下个周五
        next_dates_mask = available_dates <= next_friday
        if not next_dates_mask.any():
            continue
        actual_next_date = available_dates[next_dates_mask][-1]

        # 确保两个日期不同且都存在
        if actual_current_date >= actual_next_date:
            continue

        # 检查日期是否确实在数据中
        if actual_current_date not in available_dates or actual_next_date not in available_dates:
            continue

        try:
            # 计算收益率
            start_price = close_pivot.loc[actual_current_date]
            end_price = close_pivot.loc[actual_next_date]

            # 计算收益率，处理缺失值和无穷值
            returns = (end_price / start_price - 1).replace([np.inf, -np.inf], np.nan)

            # 构建标签数据
            for stock in returns.index:
                if not pd.isna(returns[stock]) and not pd.isna(start_price[stock]):
                    weekly_returns.append({
                        'date': current_friday,  # 使用原始的采样日期作为标识
                        'stock': stock,
                        'next_week_return': returns[stock],
                        'actual_start_date': actual_current_date,
                        'actual_end_date': actual_next_date
                    })

        except KeyError as e:
            print(f"跳过日期 {actual_current_date} 或 {actual_next_date}，原因: {e}")
            continue

    # 转换为DataFrame
    if len(weekly_returns) == 0:
        raise ValueError("未生成任何标签数据，请检查数据完整性")

    label_df = pd.DataFrame(weekly_returns)

    # 只保留必要的列用于建模
    label_df = label_df[['date', 'stock', 'next_week_return']].set_index(['date', 'stock'])

    if save_path:
        label_df.to_parquet(save_path)
        print(f"标签已保存至: {save_path}")
        print(f"标签数量: {len(label_df)}")
        print(f"标签统计:\n{label_df['next_week_return'].describe()}")

        # 打印一些统计信息
        dates_with_labels = label_df.index.get_level_values('date').unique()
        print(f"包含标签的日期数量: {len(dates_with_labels)}")
        print(f"平均每个日期的股票数量: {len(label_df) / len(dates_with_labels):.1f}")

    return label_df


def main():
    """主函数 - 生成并保存标签"""
    # 加载数据
    print("加载日频价格数据...")
    daily_df = pd.read_parquet(CFG.price_daily_file)

    print(f"价格数据形状: {daily_df.shape}")
    print(f"价格数据索引级别: {daily_df.index.names}")

    # 加载交易日历
    calendar = load_trading_calendar(CFG.trading_day_file)
    print(f"交易日历范围: {calendar.min()} 到 {calendar.max()}")

    # 获取每周最后交易日
    sample_dates = week_last_trading_days(calendar)
    print(f"采样日期数量: {len(sample_dates)}")

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