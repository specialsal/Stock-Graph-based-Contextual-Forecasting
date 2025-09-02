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


def read_csv_with_encoding(file_path: Path, **kwargs) -> pd.DataFrame:
    """
    尝试多种编码读取CSV文件
    """
    encodings = ['gbk', 'utf-8', 'gb2312', 'gb18030', 'latin-1']

    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding, **kwargs)
            print(f"成功使用 {encoding} 编码读取文件: {file_path.name}")
            return df
        except UnicodeDecodeError:
            continue

    raise ValueError(f"无法读取文件 {file_path}，尝试了所有常见编码格式")


def parse_date_flexible(date_series: pd.Series, file_name: str = "") -> pd.Series:
    """
    灵活解析日期，支持多种格式
    """
    # 先尝试常见格式
    formats = ['%Y-%m-%d', '%Y/%m/%d', '%Y%m%d', '%d/%m/%Y', '%m/%d/%Y']

    for fmt in formats:
        try:
            result = pd.to_datetime(date_series, format=fmt)
            print(f"{file_name} 使用日期格式: {fmt}")
            return result
        except ValueError:
            continue

    # 如果都失败，使用pandas的自动推断
    try:
        result = pd.to_datetime(date_series, infer_datetime_format=True)
        print(f"{file_name} 使用自动推断日期格式")
        return result
    except:
        print(f"{file_name} 无法解析日期格式，尝试使用 format='mixed'")
        return pd.to_datetime(date_series, format='mixed')


def load_stock_info() -> pd.DataFrame:
    """加载股票基础信息"""
    df = read_csv_with_encoding(CFG.stock_info_file)

    print(f"加载股票信息: {len(df)} 条记录")
    print(f"列名: {df.columns.tolist()}")

    # 解析日期 - 根据你的数据样本，使用 YYYY-MM-DD 格式
    df['ipo_date'] = parse_date_flexible(df['ipo_date'], "股票信息IPO日期")
    df['delist_date'] = parse_date_flexible(df['delist_date'], "股票信息退市日期")

    # 处理未退市股票 - 2200-01-01 表示未退市
    df.loc[df['delist_date'] > pd.Timestamp('2100-01-01'), 'delist_date'] = pd.NaT

    print(f"IPO日期范围: {df['ipo_date'].min()} 到 {df['ipo_date'].max()}")
    print(f"有效退市记录: {df['delist_date'].notna().sum()} 条")

    return df.set_index('code')


def load_suspension_data() -> pd.DataFrame:
    """加载停牌数据"""
    df = read_csv_with_encoding(CFG.is_suspended_file)

    print(f"停牌数据原始形状: {df.shape}")
    print(f"停牌数据前几行第一列样本: {df.iloc[:5, 0].tolist()}")

    # 第一列是日期，使用灵活解析
    df['date'] = parse_date_flexible(df.iloc[:, 0], "停牌数据")
    df = df.set_index('date')

    # 将TRUE/FALSE转换为1/0
    stock_cols = [col for col in df.columns if col != 'date']  # 排除日期列
    for col in stock_cols:
        df[col] = df[col].map({'TRUE': 1, 'FALSE': 0, True: 1, False: 0, 1: 1, 0: 0}).fillna(0)

    print(f"加载停牌数据: {len(df)} 个日期, {len(stock_cols)} 只股票")
    print(f"停牌数据日期范围: {df.index.min()} 到 {df.index.max()}")
    return df


def load_st_data() -> pd.DataFrame:
    """加载ST数据"""
    df = read_csv_with_encoding(CFG.is_st_file)

    print(f"ST数据原始形状: {df.shape}")
    print(f"ST数据前几行第一列样本: {df.iloc[:5, 0].tolist()}")

    # 第一列是日期，使用灵活解析
    df['date'] = parse_date_flexible(df.iloc[:, 0], "ST数据")
    df = df.set_index('date')

    # 将TRUE/FALSE转换为1/0
    stock_cols = [col for col in df.columns if col != 'date']  # 排除日期列
    for col in stock_cols:
        df[col] = df[col].map({'TRUE': 1, 'FALSE': 0, True: 1, False: 0, 1: 1, 0: 0}).fillna(0)

    print(f"加载ST数据: {len(df)} 个日期, {len(stock_cols)} 只股票")
    print(f"ST数据日期范围: {df.index.min()} 到 {df.index.max()}")
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
    total_filtered_stats = {
        'ipo_filter': 0,
        'delist_filter': 0,
        'suspension_filter': 0,
        'st_filter': 0,
        'volume_filter': 0,
        'no_data_filter': 0
    }

    for date in tqdm(sample_dates, desc="构建股票池"):
        # 获取当日所有股票
        try:
            if date in daily_df.index.get_level_values('date'):
                stocks = daily_df.loc[pd.IndexSlice[:, date], :].index.get_level_values('order_book_id').unique()
            else:
                continue
        except:
            continue

        valid_stocks = []
        date_stats = {
            'total': len(stocks),
            'ipo_filter': 0,
            'delist_filter': 0,
            'suspension_filter': 0,
            'st_filter': 0,
            'volume_filter': 0,
            'no_data_filter': 0
        }

        for stock in stocks:
            # 检查基础信息
            if stock not in stock_info.index:
                date_stats['no_data_filter'] += 1
                continue

            info = stock_info.loc[stock]

            # 检查上市时间
            days_since_ipo = (date - info['ipo_date']).days
            if days_since_ipo < CFG.ipo_cut:
                date_stats['ipo_filter'] += 1
                continue

            # 检查退市
            if pd.notna(info['delist_date']) and date >= info['delist_date']:
                date_stats['delist_filter'] += 1
                continue

            # 股票代码格式转换 (XSHE/XSHG -> XS/XSH)
            stock_code = stock.replace('.XSHE', '.XS').replace('.XSHG', '.XSH')

            # 检查停牌
            if CFG.suspended_exclude and date in suspension_df.index:
                if stock_code in suspension_df.columns:
                    if suspension_df.loc[date, stock_code] == 1:
                        date_stats['suspension_filter'] += 1
                        continue

            # 检查ST
            if CFG.st_exclude and date in st_df.index:
                if stock_code in st_df.columns:
                    if st_df.loc[date, stock_code] == 1:
                        date_stats['st_filter'] += 1
                        continue

            # 检查成交额
            try:
                if (stock, date) in daily_df.index:
                    volume = daily_df.loc[(stock, date), 'total_turnover']
                    if pd.isna(volume) or volume < CFG.min_daily_volume:
                        date_stats['volume_filter'] += 1
                        continue
                else:
                    date_stats['no_data_filter'] += 1
                    continue
            except:
                date_stats['no_data_filter'] += 1
                continue

            valid_stocks.append(stock)

        universe[date] = valid_stocks

        # 累计统计
        for key in total_filtered_stats:
            total_filtered_stats[key] += date_stats[key]

    # 打印过滤统计
    print("股票池构建统计:")
    print(f"IPO时间过滤: {total_filtered_stats['ipo_filter']}")
    print(f"退市过滤: {total_filtered_stats['delist_filter']}")
    print(f"停牌过滤: {total_filtered_stats['suspension_filter']}")
    print(f"ST过滤: {total_filtered_stats['st_filter']}")
    print(f"成交额过滤: {total_filtered_stats['volume_filter']}")
    print(f"无数据过滤: {total_filtered_stats['no_data_filter']}")

    return universe


def extract_features(
        dates: List[pd.Timestamp],
        universe: Dict[pd.Timestamp, List[str]],
        daily_df: pd.DataFrame,
        min30_df: pd.DataFrame,
        save_path: Path
):
    """
    提取并保存特征
    修复点：
        1. 不再把股票列表写入 attribute（64 KB 限制），
           改写为数据集 date_group['stocks']。
        2. stocks 数据集使用可变长 UTF-8 字符串 dtype，支持任意长度代码。
    """
    import h5py
    import numpy as np

    # HDF5 可变长字符串 dtype
    str_dtype = h5py.string_dtype(encoding='utf-8')

    with h5py.File(save_path, 'w') as h5f:
        # 保存全部采样日期（字符串列表）
        h5f.attrs['dates'] = [d.strftime('%Y-%m-%d') for d in dates]

        # 遍历每个采样日
        for date_idx, date in enumerate(tqdm(dates, desc=f"提取特征 -> {save_path.name}")):
            stocks = universe.get(date, [])
            if len(stocks) == 0:
                continue

            # 创建分组
            date_group = h5f.create_group(f'date_{date_idx}')
            date_group.attrs['date'] = date.strftime('%Y-%m-%d')

            # --- 将股票列表写成数据集，而非 attribute ---
            date_group.create_dataset(
                'stocks',
                data=np.asarray(stocks, dtype=str_dtype),
                compression='gzip'          # 可选：开启压缩
            )

            # ———————— 日频特征 ————————
            daily_features = []
            for stock in stocks:
                d_end = date
                d_start = date - pd.Timedelta(days=CFG.daily_window + 30)  # 取宽窗口

                try:
                    stock_data = daily_df.loc[stock]
                    if isinstance(stock_data, pd.Series):
                        stock_data = stock_data.to_frame().T

                    mask = (stock_data.index >= d_start) & (stock_data.index <= d_end)
                    daily_data = stock_data.loc[mask]

                    if len(daily_data) >= CFG.daily_window:
                        daily_data = daily_data.tail(CFG.daily_window)
                        feats = daily_data[['open', 'high', 'low', 'close', 'volume', 'num_trades']].values
                        daily_features.append(feats)
                    else:
                        daily_features.append(np.full((CFG.daily_window, 6), np.nan))
                except Exception:
                    daily_features.append(np.full((CFG.daily_window, 6), np.nan))

            if daily_features:
                date_group.create_dataset(
                    'daily',
                    data=np.array(daily_features, dtype=np.float32),
                    compression='gzip'
                )

            # ———————— 30分钟特征 ————————
            min30_features = []
            for stock in stocks:
                m_end = pd.Timestamp(date.strftime('%Y-%m-%d 15:00:00'))
                m_start = m_end - pd.Timedelta(minutes=30 * (CFG.min30_window + 20))

                try:
                    stock_data = min30_df.loc[stock]
                    if isinstance(stock_data, pd.Series):
                        stock_data = stock_data.to_frame().T

                    mask = (stock_data.index >= m_start) & (stock_data.index <= m_end)
                    min_data = stock_data.loc[mask]

                    if len(min_data) >= CFG.min30_window:
                        min_data = min_data.tail(CFG.min30_window)
                        feats = min_data[['open', 'high', 'low', 'close', 'volume', 'total_turnover']].values
                        min30_features.append(feats)
                    else:
                        min30_features.append(np.full((CFG.min30_window, 6), np.nan))
                except Exception:
                    min30_features.append(np.full((CFG.min30_window, 6), np.nan))

            if min30_features:
                date_group.create_dataset(
                    'min30',
                    data=np.array(min30_features, dtype=np.float32),
                    compression='gzip'
                )

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
        print(f"日频Scaler拟合完成，样本数: {len(all_daily)}")

    if min30_samples:
        all_min30 = np.concatenate(min30_samples, axis=0)
        min30_scaler.fit(all_min30)
        print(f"30分钟Scaler拟合完成，样本数: {len(all_min30)}")

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

    print(f"日频数据形状: {daily_df.shape}")
    print(f"30分钟数据形状: {min30_df.shape}")

    # 加载交易日历
    calendar = load_trading_calendar(CFG.trading_day_file)
    sample_dates = week_last_trading_days(calendar)

    print(f"交易日历范围: {calendar.min()} 到 {calendar.max()}")
    print(f"采样日期数量: {len(sample_dates)}")

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

    # 统计股票池信息
    total_samples = sum(len(stocks) for stocks in universe.values())
    avg_stocks_per_date = total_samples / len(universe) if universe else 0
    print(f"总样本数: {total_samples}")
    print(f"平均每日股票数: {avg_stocks_per_date:.1f}")

    # 划分数据集
    train_dates = [d for d in sample_dates if d.year <= CFG.train_end_year]
    val_dates = [d for d in sample_dates if d.year == CFG.val_end_year]
    test_dates = [d for d in sample_dates if d.year > CFG.val_end_year]

    print(f"训练集: {len(train_dates)} 个日期 ({train_dates[0]} 到 {train_dates[-1]})")
    print(f"验证集: {len(val_dates)} 个日期 ({val_dates[0]} 到 {val_dates[-1]})")
    print(f"测试集: {len(test_dates)} 个日期 ({test_dates[0]} 到 {test_dates[-1]})")

    # 提取特征
    print("提取特征...")
    extract_features(train_dates, universe, daily_df, min30_df, CFG.train_features_file)
    extract_features(val_dates, universe, daily_df, min30_df, CFG.val_features_file)
    if test_dates:
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
    print("数据预处理完成！")


if __name__ == "__main__":
    main()