# coding: utf-8
"""
数据预处理模块 - 构建特征和股票池
"""
import pandas as pd
import numpy as np
import h5py
import pickle
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple
from config import CFG
from utils import (
    load_trading_calendar,
    week_last_trading_days,
    mad_clip,
    Scaler,
    save_universe
)


def load_stock_info() -> pd.DataFrame:
    """加载股票基础信息"""
    df = pd.read_csv(CFG.stock_info_file)
    df['ipo_date'] = pd.to_datetime(df['ipo_date'], format='%Y/%m/%d')
    df['delist_date'] = pd.to_datetime(df['delist_date'], format='%Y/%m/%d')
    # 处理未退市股票
    df.loc[df['delist_date'] > pd.Timestamp('2100-01-01'), 'delist_date'] = pd.NaT
    return df.set_index('code')


def load_suspension_data() -> pd.DataFrame:
    """加载停牌数据"""
    df = pd.read_csv(CFG.is_suspended_file)
    df['date'] = pd.to_datetime(df.iloc[:, 0], format='%Y/%m/%d')
    df = df.set_index('date')
    # 将TRUE/FALSE转换为1/0
    stock_cols = df.columns[1:]
    for col in stock_cols:
        df[col] = df[col].map({'TRUE': 1, 'FALSE': 0, True: 1, False: 0})
    return df


def load_st_data() -> pd.DataFrame:
    """加载ST数据"""
    df = pd.read_csv(CFG.is_st_file)
    df['date'] = pd.to_datetime(df.iloc[:, 0], format='%Y/%m/%d')
    df = df.set_index('date')
    # 将TRUE/FALSE转换为1/0
    stock_cols = df.columns[1:]
    for col in stock_cols:
        df[col] = df[col].map({'TRUE': 1, 'FALSE': 0, True: 1, False: 0})
    return df


def build_universe(
        sample_dates: pd.DatetimeIndex,
        daily_df: pd.DataFrame,
        stock_info: pd.DataFrame,
        suspension_df: pd.DataFrame,
        st_df: pd.DataFrame
) -> Dict[pd.Timestamp, List[str]]:
    """
    构建股票池
    """
    universe = {}

    for date in tqdm(sample_dates, desc="构建股票池"):
        # 获取当日所有股票
        if date in daily_df.index.get_level_values('date'):
            stocks = daily_df.loc[pd.IndexSlice[:, date], :].index.get_level_values('order_book_id').unique()
        else:
            continue

        valid_stocks = []

        for stock in stocks:
            # 股票代码格式转换 (XSHE/XSHG -> XS/XSH)
            stock_code = stock.replace('.XSHE', '.XS').replace('.XSHG', '.XSH')

            # 检查基础信息
            if stock not in stock_info.index:
                continue

            info = stock_info.loc[stock]

            # 检查上市时间
            days_since_ipo = (date - info['ipo_date']).days
            if days_since_ipo < CFG.ipo_cut:
                continue

            # 检查退市
            if pd.notna(info['delist_date']) and date >= info['delist_date']:
                continue

            # 检查停牌
            if CFG.suspended_exclude and date in suspension_df.index:
                if stock_code in suspension_df.columns:
                    if suspension_df.loc[date, stock_code] == 1:
                        continue

            # 检查ST
            if CFG.st_exclude and date in st_df.index:
                if stock_code in st_df.columns:
                    if st_df.loc[date, stock_code] == 1:
                        continue

            # 检查成交额
            if (stock, date) in daily_df.index:
                volume = daily_df.loc[(stock, date), 'total_turnover']
                if volume < CFG.min_daily_volume:
                    continue

            valid_stocks.append(stock)

        universe[date] = valid_stocks

    return universe


def extract_features(
        dates: List[pd.Timestamp],
        universe: Dict[pd.Timestamp, List[str]],
        daily_df: pd.DataFrame,
        min30_df: pd.DataFrame,
        save_path: Path
):
    """提取并保存特征"""

    with h5py.File(save_path, 'w') as h5f:
        # 存储元数据
        h5f.attrs['dates'] = [d.strftime('%Y-%m-%d') for d in dates]

        for date_idx, date in enumerate(tqdm(dates, desc=f"提取特征 -> {save_path.name}")):
            stocks = universe.get(date, [])
            if len(stocks) == 0:
                continue

            date_group = h5f.create_group(f'date_{date_idx}')
            date_group.attrs['date'] = date.strftime('%Y-%m-%d')
            date_group.attrs['stocks'] = stocks

            # 提取日频特征
            daily_features = []
            for stock in stocks:
                # 日频窗口
                d_end = date
                d_start = date - pd.Timedelta(days=CFG.daily_window + 30)  # 多取一些以保证有足够数据

                if (stock, d_start) in daily_df.index:
                    daily_data = daily_df.loc[pd.IndexSlice[stock, d_start:d_end], :]
                    # 确保有足够的数据点
                    if len(daily_data) >= CFG.daily_window:
                        daily_data = daily_data.tail(CFG.daily_window)
                        features = daily_data[['open', 'high', 'low', 'close', 'volume', 'total_turnover']].values
                        daily_features.append(features)
                    else:
                        # 填充nan
                        daily_features.append(np.full((CFG.daily_window, 6), np.nan))
                else:
                    daily_features.append(np.full((CFG.daily_window, 6), np.nan))

            if daily_features:
                date_group.create_dataset('daily', data=np.array(daily_features))

            # 提取30分钟特征
            min30_features = []
            for stock in stocks:
                # 30分钟窗口 - 当天15:00往前推80个bar
                m_end = pd.Timestamp(date.strftime('%Y-%m-%d 15:00:00'))
                m_start = m_end - pd.Timedelta(minutes=30 * (CFG.min30_window + 20))  # 多取一些

                if (stock, m_start) in min30_df.index:
                    min_data = min30_df.loc[pd.IndexSlice[stock, m_start:m_end], :]
                    if len(min_data) >= CFG.min30_window:
                        min_data = min_data.tail(CFG.min30_window)
                        features = min_data[['open', 'high', 'low', 'close', 'volume', 'total_turnover']].values
                        min30_features.append(features)
                    else:
                        min30_features.append(np.full((CFG.min30_window, 6), np.nan))
                else:
                    min30_features.append(np.full((CFG.min30_window, 6), np.nan))

            if min30_features:
                date_group.create_dataset('min30', data=np.array(min30_features))


def fit_scalers(h5_path: Path) -> Tuple[Scaler, Scaler]:
    """在训练集上拟合标准化器"""
    daily_scaler = Scaler()
    min30_scaler = Scaler()

    daily_samples = []
    min30_samples = []

    with h5py.File(h5_path, 'r') as h5f:
        for key in tqdm(h5f.keys(), desc="拟合Scaler"):
            if 'daily' in h5f[key]:
                data = h5f[key]['daily'][:]
                # 去除nan
                valid_data = data[~np.isnan(data).any(axis=(1, 2))]
                if len(valid_data) > 0:
                    daily_samples.append(valid_data)

            if 'min30' in h5f[key]:
                data = h5f[key]['min30'][:]
                valid_data = data[~np.isnan(data).any(axis=(1, 2))]
                if len(valid_data) > 0:
                    min30_samples.append(valid_data)

    if daily_samples:
        all_daily = np.concatenate(daily_samples, axis=0)
        daily_scaler.fit(all_daily)

    if min30_samples:
        all_min30 = np.concatenate(min30_samples, axis=0)
        min30_scaler.fit(all_min30)

    return daily_scaler, min30_scaler


def main():
    """主函数"""
    # 加载数据
    print("加载数据...")
    daily_df = pd.read_parquet(CFG.price_daily_file)
    min30_df = pd.read_parquet(CFG.price_30m_file)
    stock_info = load_stock_info()
    suspension_df = load_suspension_data()
    st_df = load_st_data()

    # 加载交易日历
    calendar = load_trading_calendar(CFG.trading_day_file)
    sample_dates = week_last_trading_days(calendar)

    # 构建股票池
    print("构建股票池...")
    universe = build_universe(
        sample_dates,
        daily_df,
        stock_info,
        suspension_df,
        st_df
    )

    # 保存股票池
    save_universe(universe, CFG.universe_file)
    print(f"股票池已保存至: {CFG.universe_file}")

    # 划分数据集
    train_dates = [d for d in sample_dates if d.year <= CFG.train_end_year]
    val_dates = [d for d in sample_dates if d.year == CFG.val_end_year]
    test_dates = [d for d in sample_dates if d.year > CFG.val_end_year]

    print(f"训练集: {len(train_dates)} 个日期")
    print(f"验证集: {len(val_dates)} 个日期")
    print(f"测试集: {len(test_dates)} 个日期")

    # 提取特征
    extract_features(train_dates, universe, daily_df, min30_df, CFG.train_features_file)
    extract_features(val_dates, universe, daily_df, min30_df, CFG.val_features_file)
    extract_features(test_dates, universe, daily_df, min30_df, CFG.test_features_file)

    # 拟合并保存Scaler
    print("拟合标准化器...")
    daily_scaler, min30_scaler = fit_scalers(CFG.train_features_file)

    with open(CFG.scaler_file, 'wb') as f:
        pickle.dump({
            'daily': daily_scaler,
            'min30': min30_scaler
        }, f)
    print(f"Scaler已保存至: {CFG.scaler_file}")


if __name__ == "__main__":
    main()