# -*- coding: utf-8 -*-
import os
import warnings
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
import rqdatac as rq
import akshare as ak

# =========================
# 基础配置
# =========================
# 如果有账号密码: rq.init('user','pwd')
rq.init()

DATA_PATH = 'data/raw/'
os.makedirs(DATA_PATH, exist_ok=True)

TODAY_STR = datetime.today().strftime('%Y-%m-%d')
FREQ_DAY = '1d'
TRADING_DAY_END = '2030-12-31'

# 是否更新的开关（可按需修改）
UPDATE_SWITCH = {
    'trading_calendar': True,      # 交易日 & 交易周
    'stock_info': True,            # 股票信息（聚宽导出覆盖）
    'stock_price_day': True,       # 股票日行情（parquet, MultiIndex）
    'stock_fundamental_day': False, # 股票基本面日行情（parquet, MultiIndex）
    'suspended': True,             # 停牌（CSV 宽表）
    'is_st': True,                 # ST（CSV 宽表）
    'index_components': True,      # 指数成分（快照覆盖）
    'industry_and_style': True,    # 行业与风格（快照覆盖 + 映射）
    'index_price_day': True,       # 指数日行情（parquet, MultiIndex）
    'sector_price_day': True,      # 新增：风格日行情（增量聚合）
}

# 可选：在获取行情时过滤无效代码，减少 invalid order_book_id 警告
FILTER_INVALID_CODES = True
FACTOR_LIST = ['pe_ratio_ttm','book_to_market_ratio_ttm','ps_ratio_ttm','market_cap_2','dividend_yield_ttm','cash_flow_per_share_ttm','inc_revenue_ttm','net_profit_growth_ratio_ttm'] # 基本面因子列表
# 市盈率、市净率、市销率、流通股总市值、股息率、每股经营现金流、营业收入同比增长率、净利润同比增长率
# =========================
# 通用工具
# =========================
def to_date_str(d):
    if isinstance(d, (pd.Timestamp, datetime, np.datetime64)):
        return pd.to_datetime(d).strftime('%Y-%m-%d')
    if isinstance(d, str):
        return d[:10]
    return str(d)

def next_day_str(date_str):
    return (pd.to_datetime(date_str) + timedelta(days=1)).strftime('%Y-%m-%d')

def read_csv_safe(path, **kwargs):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, **kwargs)

def read_parquet_safe(path):
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)

def write_csv(df, path, **kwargs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, **kwargs)

def write_parquet(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path)

def max_date_from_index_as_str(df):
    # 适用于 index 为日期（DatetimeIndex 或可转日期的字符串）
    if df is None or df.empty:
        return None
    try:
        return pd.to_datetime(df.index).max().strftime('%Y-%m-%d')
    except Exception:
        return None

def max_datetime_from_multiindex(df):
    # 适用于 MultiIndex(levels: order_book_id, datetime)
    if df is None or df.empty or not isinstance(df.index, pd.MultiIndex):
        return None
    if 'datetime' in df.index.names:
        dt = df.index.get_level_values('datetime')
    else:
        # 兼容第二层
        dt = df.index.get_level_values(1)
    return pd.to_datetime(dt).max()

def align_columns_union(df_old, df_new):
    # 行情类列名可能不一致，做并集对齐
    cols = sorted(set(df_old.columns) | set(df_new.columns))
    df_old_aligned = df_old.reindex(columns=cols)
    df_new_aligned = df_new.reindex(columns=cols)
    return df_old_aligned, df_new_aligned

def concat_dedup_multiindex(df_old, df_new, sort_index=True):
    # 专用于 MultiIndex(order_book_id, datetime) 的增量合并
    if df_old is None or df_old.empty:
        df = df_new.copy()
    elif df_new is None or df_new.empty:
        df = df_old.copy()
    else:
        # 对齐列
        df_old2, df_new2 = align_columns_union(df_old, df_new)
        df = pd.concat([df_old2, df_new2], axis=0)
        # 按索引去重，保留最后
        df = df[~df.index.duplicated(keep='last')]
    if sort_index and df is not None and not df.empty:
        df = df.sort_index()
    return df

def concat_dedup_wide_by_index(df_old, df_new, sort_index=True):
    # 停牌 / ST 宽表：index=日期，columns=股票代码
    if df_old is None or df_old.empty:
        df = df_new.copy()
    elif df_new is None or df_new.empty:
        df = df_old.copy()
    else:
        cols = sorted(set(df_old.columns) | set(df_new.columns))
        idx = sorted(set(pd.to_datetime(df_old.index)) | set(pd.to_datetime(df_new.index)))
        df_old2 = df_old.copy()
        df_old2.index = pd.to_datetime(df_old2.index)
        df_new2 = df_new.copy()
        df_new2.index = pd.to_datetime(df_new2.index)
        df_old2 = df_old2.reindex(index=idx, columns=cols)
        df_new2 = df_new2.reindex(index=idx, columns=cols)
        # 新数据优先覆盖旧数据
        df = df_old2.combine_first(df_new2)
        df = df_new2.combine_first(df)
    if sort_index and df is not None and not df.empty:
        df = df.sort_index()
    return df

def get_price_safe(codes, start_date, end_date, frequency='1d', fields=None,
                   adjust_type='pre', skip_suspended=False, market='cn',
                   expect_df=True, time_slice=None, filter_codes=True):
    code_list = list(codes)
    if filter_codes and FILTER_INVALID_CODES:
        if len(code_list) == 0:
            return None
    try:
        df = rq.get_price(
            code_list, start_date=start_date, end_date=end_date,
            frequency=frequency, fields=fields, adjust_type=adjust_type,
            skip_suspended=skip_suspended, market=market,
            expect_df=expect_df, time_slice=time_slice
        )
        if df is None:
            return None
        if isinstance(df, pd.DataFrame) and df.empty:
            return None
        return df
    except Exception as e:
        warnings.warn(f'get_price_safe error: {e}')
        return None
    
def get_factor_safe(codes, factor, start_date, end_date, universe=None, market='cn',
                   expect_df=True, filter_codes=True):
    code_list = list(codes)
    if filter_codes and FILTER_INVALID_CODES:
        if len(code_list) == 0:
            return None
    try:
        df = rq.get_factor(code_list, 
                           factor, 
                           start_date=start_date, 
                           end_date=end_date, 
                           universe=universe,
                           expect_df=expect_df, 
                           market=market)
        if df is None:
            return None
        if isinstance(df, pd.DataFrame) and df.empty:
            return None
        return df
    except Exception as e:
        warnings.warn(f'get_factor_safe error: {e}')
        return None

# =========================
# 1) 交易日与交易周（增量）
# =========================
def update_trading_calendar(start='2001-01-01', end=TRADING_DAY_END):
    if not UPDATE_SWITCH.get('trading_calendar', True):
        return
    # 交易日
    path_day = os.path.join(DATA_PATH, 'trading_day.csv')
    df_old = read_csv_safe(path_day, index_col=0)
    if df_old is not None:
        # 兼容旧格式
        if 'trading_day' not in df_old.columns and df_old.shape[1] == 1:
            df_old.columns = ['trading_day']
    last_day = None
    if df_old is not None and not df_old.empty:
        last_day = pd.to_datetime(df_old['trading_day']).max().strftime('%Y-%m-%d')

    # 改为包含 last_day 本身
    fetch_start = start if last_day is None else last_day
    if pd.to_datetime(fetch_start) <= pd.to_datetime(end):
        new_days = rq.get_trading_dates(start_date=fetch_start.replace('-', ''), end_date=end.replace('-', ''))
        new_days = [d.strftime('%Y-%m-%d') for d in new_days]
        df_new = pd.DataFrame(new_days, columns=['trading_day'])
    else:
        df_new = pd.DataFrame(columns=['trading_day'])

    df_day = pd.concat([df_old[['trading_day']] if df_old is not None else pd.DataFrame(columns=['trading_day']),
                        df_new], ignore_index=True)
    df_day = df_day.drop_duplicates().sort_values('trading_day').reset_index(drop=True)
    write_csv(df_day, path_day)

    # 交易周（根据交易日计算）：当日与下一交易日不相邻（相差 > 1 天）视为周末
    days = pd.to_datetime(df_day['trading_day']).tolist()
    weeks = [d for i, d in enumerate(days[:-1]) if (days[i+1] - days[i]).days > 1]
    df_week = pd.DataFrame(pd.Series(weeks).dt.strftime('%Y-%m-%d'), columns=['trading_week'])
    path_week = os.path.join(DATA_PATH, 'trading_week.csv')
    write_csv(df_week, path_week)


# =========================
# 2) 全部股票信息（直接由聚宽导出覆盖，不做增量）
# =========================
def update_stock_info_from_jq_export():
    if not UPDATE_SWITCH.get('stock_info', True):
        return
    src_path = os.path.join(DATA_PATH, 'stock_list_jq.csv')
    if not os.path.exists(src_path):
        warnings.warn('缺少 data/raw/stock_list_jq.csv，跳过 stock_info 更新')
        return
    df_src = pd.read_csv(src_path, index_col=0)
    df_src = df_src.reset_index().rename(columns={
        'index': 'code',
        'display_name': 'name',
        'start_date': 'ipo_date',
        'end_date': 'delist_date'
    })
    df_src = df_src[['code', 'name', 'ipo_date', 'delist_date']]
    out_path = os.path.join(DATA_PATH, 'stock_info.csv')
    write_csv(df_src, out_path, encoding='gbk')


# =========================
# 3) 股票日行情（前复权，MultiIndex 增量）
# =========================
def update_stock_price_day(start='2010-01-01', end=TODAY_STR):
    if not UPDATE_SWITCH.get('stock_price_day', True):
        return
    stock_info_path = os.path.join(DATA_PATH, 'stock_info.csv')
    stock_info = read_csv_safe(stock_info_path, index_col=0, encoding='gbk')
    if stock_info is None or stock_info.empty:
        warnings.warn('stock_info.csv 不存在或为空，跳过行情更新')
        return
    stock_list = stock_info['code'].dropna().unique().tolist()

    out_path = os.path.join(DATA_PATH, 'stock_price_day.parquet')
    df_old = read_parquet_safe(out_path)

    # 确保旧数据为 MultiIndex（order_book_id, datetime）
    if df_old is not None and not df_old.empty:
        if not isinstance(df_old.index, pd.MultiIndex):
            idx_cols = [c for c in df_old.columns if c in ['order_book_id', 'datetime']]
            if set(idx_cols) == {'order_book_id', 'datetime'}:
                df_old['datetime'] = pd.to_datetime(df_old['datetime'])
                df_old = df_old.set_index(['order_book_id', 'datetime']).sort_index()

    last_dt = max_datetime_from_multiindex(df_old)
    # 改为包含 last_dt 当天
    fetch_start = start if last_dt is None else last_dt.strftime('%Y-%m-%d')
    if pd.to_datetime(fetch_start) > pd.to_datetime(end):
        return

    df_new = get_price_safe(
        stock_list,
        start_date=fetch_start,
        end_date=end,
        frequency=FREQ_DAY,
        fields=None,
        adjust_type='pre',
        skip_suspended=False,
        market='cn',
        expect_df=True,
        time_slice=None,
        filter_codes=True
    )
    # 若无增量数据（当天无数据或全部非法），直接退出
    if df_new is None:
        return

    # 统一转 MultiIndex
    if isinstance(df_new.index, pd.MultiIndex):
        df_new_mi = df_new.copy()
    else:
        if {'order_book_id', 'datetime'}.issubset(df_new.columns):
            df_new['datetime'] = pd.to_datetime(df_new['datetime'])
            df_new_mi = df_new.set_index(['order_book_id', 'datetime']).sort_index()
        else:
            df_new_mi = df_new.reset_index().set_index(['order_book_id', 'datetime']).sort_index()

    # 合并：列对齐并去重
    df_all = concat_dedup_multiindex(df_old, df_new_mi, sort_index=True)
    write_parquet(df_all, out_path)

# =========================
# 3.5) 股票基本面行情（前复权，MultiIndex 增量）
# =========================
def update_stock_fundamental_day(start='2010-01-01', end=TODAY_STR,factor_list=FACTOR_LIST):
    if not UPDATE_SWITCH.get('stock_fundamental_day', True):
        return
    stock_info_path = os.path.join(DATA_PATH, 'stock_info.csv')
    stock_info = read_csv_safe(stock_info_path, index_col=0, encoding='gbk')
    if stock_info is None or stock_info.empty:
        warnings.warn('stock_info.csv 不存在或为空，跳过行情更新')
        return
    stock_list = stock_info['code'].dropna().unique().tolist()

    out_path = os.path.join(DATA_PATH, 'stock_fundamental_day.parquet')
    df_old = read_parquet_safe(out_path)

    # 确保旧数据为 MultiIndex（order_book_id, datetime）
    if df_old is not None and not df_old.empty:
        if not isinstance(df_old.index, pd.MultiIndex):
            idx_cols = [c for c in df_old.columns if c in ['order_book_id', 'date']]
            if set(idx_cols) == {'order_book_id', 'date'}:
                df_old['date'] = pd.to_datetime(df_old['date'])
                df_old = df_old.set_index(['order_book_id', 'date']).sort_index()

    last_dt = max_datetime_from_multiindex(df_old)
    # 改为包含 last_dt 当天
    fetch_start = start if last_dt is None else last_dt.strftime('%Y-%m-%d')
    if pd.to_datetime(fetch_start) > pd.to_datetime(end):
        return

    df_new = get_factor_safe(
        stock_list,
        factor = factor_list,
        start_date=fetch_start,
        end_date=end,
        universe=None,
        market='cn',
        expect_df=True,
        filter_codes=True
)
    # 若无增量数据（当天无数据或全部非法），直接退出
    if df_new is None:
        return

    # 统一转 MultiIndex
    if isinstance(df_new.index, pd.MultiIndex):
        df_new_mi = df_new.copy()
    else:
        if {'order_book_id', 'date'}.issubset(df_new.columns):
            df_new['date'] = pd.to_datetime(df_new['date'])
            df_new_mi = df_new.set_index(['order_book_id', 'date']).sort_index()
        else:
            df_new_mi = df_new.reset_index().set_index(['order_book_id', 'date']).sort_index()

    # 合并：列对齐并去重
    df_all = concat_dedup_multiindex(df_old, df_new_mi, sort_index=True)
    write_parquet(df_all, out_path)
# =========================
# 4) 停牌信息（宽表，增量）
# =========================
def update_suspended(start='2010-01-01', end=TODAY_STR):
    if not UPDATE_SWITCH.get('suspended', True):
        return
    stock_info_path = os.path.join(DATA_PATH, 'stock_info.csv')
    stock_info = read_csv_safe(stock_info_path, index_col=0, encoding='gbk')
    if stock_info is None or stock_info.empty:
        warnings.warn('stock_info.csv 不存在或为空，跳过停牌更新')
        return
    stock_list = stock_info['code'].dropna().unique().tolist()

    out_path = os.path.join(DATA_PATH, 'is_suspended.csv')
    df_old = read_csv_safe(out_path, index_col=0)
    if df_old is not None and not df_old.empty:
        try:
            df_old.index = pd.to_datetime(df_old.index)
        except Exception:
            pass

    last_day = max_date_from_index_as_str(df_old)
    # 改为包含 last_day 当天
    fetch_start = start if last_day is None else last_day
    if pd.to_datetime(fetch_start) > pd.to_datetime(end):
        return

    try:
        suspended = rq.is_suspended(stock_list, start_date=fetch_start, end_date=end, market='cn')
    except Exception as e:
        warnings.warn(f'is_suspended error: {e}')
        return
    if suspended is None or (isinstance(suspended, pd.DataFrame) and suspended.empty):
        return

    suspended.index = pd.to_datetime(suspended.index)
    df_all = concat_dedup_wide_by_index(df_old, suspended, sort_index=True)
    df_to_save = df_all.copy()
    df_to_save.index = df_to_save.index.strftime('%Y-%m-%d')
    write_csv(df_to_save, out_path)


# =========================
# 5) ST 信息（宽表，增量）
# =========================
def update_is_st(start='2010-01-01', end=TODAY_STR):
    if not UPDATE_SWITCH.get('is_st', True):
        return
    stock_info_path = os.path.join(DATA_PATH, 'stock_info.csv')
    stock_info = read_csv_safe(stock_info_path, index_col=0, encoding='gbk')
    if stock_info is None or stock_info.empty:
        warnings.warn('stock_info.csv 不存在或为空，跳过 ST 更新')
        return
    stock_list = stock_info['code'].dropna().unique().tolist()

    out_path = os.path.join(DATA_PATH, 'is_st_stock.csv')
    df_old = read_csv_safe(out_path, index_col=0)
    if df_old is not None and not df_old.empty:
        try:
            df_old.index = pd.to_datetime(df_old.index)
        except Exception:
            pass

    last_day = max_date_from_index_as_str(df_old)
    # 改为包含 last_day 当天
    fetch_start = start if last_day is None else last_day
    if pd.to_datetime(fetch_start) > pd.to_datetime(end):
        return

    try:
        is_st = rq.is_st_stock(stock_list, start_date=fetch_start, end_date=end)
    except Exception as e:
        warnings.warn(f'is_st_stock error: {e}')
        return
    if is_st is None or (isinstance(is_st, pd.DataFrame) and is_st.empty):
        return

    is_st.index = pd.to_datetime(is_st.index)
    df_all = concat_dedup_wide_by_index(df_old, is_st, sort_index=True)
    df_to_save = df_all.copy()
    df_to_save.index = df_to_save.index.strftime('%Y-%m-%d')
    write_csv(df_to_save, out_path)


# =========================
# 6) 指数成分（当前快照覆盖）
# =========================
def update_index_components():
    if not UPDATE_SWITCH.get('index_components', True):
        return
    indices = {
        '000300.XSHG': 'index_components_000300.csv',
        '000852.XSHG': 'index_components_000852.csv',
        '000905.XSHG': 'index_components_000905.csv',
    }
    for idx, fname in indices.items():
        comps = pd.Series(rq.index_components(idx, date=None, market='cn', return_create_tm=False))
        out_path = os.path.join(DATA_PATH, fname)
        write_csv(comps, out_path)


# =========================
# 7) 行业与风格（当前快照覆盖 + 映射）
# =========================
def update_industry_and_style():
    if not UPDATE_SWITCH.get('industry_and_style', True):
        return
    stock_info_path = os.path.join(DATA_PATH, 'stock_info.csv')
    stock_info = read_csv_safe(stock_info_path, index_col=0, encoding='gbk')
    if stock_info is None or stock_info.empty:
        warnings.warn('stock_info.csv 不存在或为空，跳过行业/风格更新')
        return
    stock_list = stock_info['code'].dropna().unique().tolist()

    # 行业（中信2019，Level 0）
    stock_industry = rq.get_instrument_industry(stock_list, source='citics_2019', level=0)
    out_ind_path = os.path.join(DATA_PATH, 'stock_industry_citics_2019_level0.csv')
    write_csv(stock_industry, out_ind_path, encoding='gbk')

    # 风格/板块（citics_sector）
    stock_sector = rq.get_instrument_industry(stock_list, source='citics_2019', level='citics_sector')
    out_style_path = os.path.join(DATA_PATH, 'stock_style_citics_2019_sector.csv')
    write_csv(stock_sector, out_style_path, encoding='gbk')

    # 生成映射（覆盖）
    pd.read_csv(out_style_path, encoding='gbk')[['order_book_id', 'style_sector_name']] \
        .rename(columns={'style_sector_name': 'sector'}) \
        .to_csv(os.path.join(DATA_PATH, 'stock_style_map.csv'), index=False, encoding='gbk')

    pd.read_csv(out_ind_path, encoding='gbk')[['order_book_id', 'first_industry_name']] \
        .rename(columns={'first_industry_name': 'industry'}) \
        .to_csv(os.path.join(DATA_PATH, 'stock_industry_map.csv'), index=False, encoding='gbk')


# =========================
# 8) 指数日行情（MultiIndex 增量）
# =========================
def update_index_price_day(start='2010-01-01', end=TODAY_STR):
    if not UPDATE_SWITCH.get('index_price_day', True):
        return
    index_list = ['000300.XSHG', '000905.XSHG', '000852.XSHG']

    out_path = os.path.join(DATA_PATH, 'index_price_day.parquet')
    df_old = read_parquet_safe(out_path)

    # 旧数据 MultiIndex 化
    if df_old is not None and not df_old.empty:
        if not isinstance(df_old.index, pd.MultiIndex):
            idx_cols = [c for c in df_old.columns if c in ['order_book_id', 'datetime']]
            if set(idx_cols) == {'order_book_id', 'datetime'}:
                df_old['datetime'] = pd.to_datetime(df_old['datetime'])
                df_old = df_old.set_index(['order_book_id', 'datetime']).sort_index()

    last_dt = max_datetime_from_multiindex(df_old)
    # 改为包含 last_dt 当天
    fetch_start = start if last_dt is None else last_dt.strftime('%Y-%m-%d')
    if pd.to_datetime(fetch_start) > pd.to_datetime(end):
        return

    df_new = get_price_safe(
        index_list,
        start_date=fetch_start,
        end_date=end,
        frequency=FREQ_DAY,
        fields=None,
        adjust_type='pre',
        skip_suspended=False,
        market='cn',
        expect_df=True,
        time_slice=None,
        filter_codes=False  # 指数代码一般无需过滤
    )
    if df_new is None:
        return

    if isinstance(df_new.index, pd.MultiIndex):
        df_new_mi = df_new.copy()
    else:
        if {'order_book_id', 'datetime'}.issubset(df_new.columns):
            df_new['datetime'] = pd.to_datetime(df_new['datetime'])
            df_new_mi = df_new.set_index(['order_book_id', 'datetime']).sort_index()
        else:
            df_new_mi = df_new.reset_index().set_index(['order_book_id', 'datetime']).sort_index()

    df_all = concat_dedup_multiindex(df_old, df_new_mi, sort_index=True)
    write_parquet(df_all, out_path)


# =========================
# 9) 新增：风格日行情（增量聚合到 sector_price_day.parquet）
# =========================
def update_sector_price_day(start='2010-01-01', end=TODAY_STR):
    if not UPDATE_SWITCH.get('sector_price_day', True):
        return

    # 依赖输入
    stock_price_path = os.path.join(DATA_PATH, 'stock_price_day.parquet')
    style_map_path   = os.path.join(DATA_PATH, 'stock_style_map.csv')

    price_df = read_parquet_safe(stock_price_path)
    if price_df is None or price_df.empty:
        warnings.warn('缺少或空的 stock_price_day.parquet，跳过 sector_price_day 生成')
        return

    # MultiIndex(order_book_id, datetime)
    if not isinstance(price_df.index, pd.MultiIndex):
        idx_cols = [c for c in price_df.columns if c in ['order_book_id', 'datetime']]
        if set(idx_cols) == {'order_book_id', 'datetime'}:
            price_df['datetime'] = pd.to_datetime(price_df['datetime'])
            price_df = price_df.set_index(['order_book_id', 'datetime']).sort_index()
        else:
            warnings.warn('stock_price_day.parquet 非 MultiIndex 结构，跳过 sector 生成')
            return

    style_map = read_csv_safe(style_map_path, encoding='gbk')
    if style_map is None or style_map.empty or ('order_book_id' not in style_map.columns) or ('sector' not in style_map.columns):
        warnings.warn('缺少 stock_style_map.csv 或列不正确，跳过 sector 生成')
        return
    style_map['order_book_id'] = style_map['order_book_id'].astype(str).str.strip()
    style_map['sector'] = style_map['sector'].astype(str).str.strip()

    # 目标风格枚举（固定五类）
    TARGET_SECTORS = ['周期风格', '成长风格', '消费风格', '稳定风格', '金融风格']

    # 取出我们需要的行情列（这里最好保证有 open, close, total_turnover）
    cols_needed = []
    for c in ['open', 'close', 'high', 'low', 'volume', 'total_turnover', 'num_trades']:
        if c in price_df.columns:
            cols_needed.append(c)
    if not cols_needed or ('close' not in cols_needed) or ('total_turnover' not in cols_needed):
        warnings.warn('price_day 缺少必要列（至少需要 close 与 total_turnover），跳过 sector 生成')
        return
    price_small = price_df[cols_needed].copy()

    # 把 MultiIndex 拆成列，便于合并风格映射
    price_small = price_small.reset_index()
    price_small['order_book_id'] = price_small['order_book_id'].astype(str).str.strip()
    # 这里 MultiIndex 第二层叫 'datetime'，要改成 'date'
    price_small = price_small.rename(columns={'datetime': 'date'})
    price_small['date'] = pd.to_datetime(price_small['date'])

    # 合并 sector
    price_small = price_small.merge(style_map[['order_book_id', 'sector']], on='order_book_id', how='left')
    price_small['sector'] = price_small['sector'].fillna('')  # 无风格的剔除
    price_small = price_small[price_small['sector'].isin(TARGET_SECTORS)]

    if price_small.empty:
        warnings.warn('按风格过滤后为空，跳过 sector 生成')
        return

    # 读取已有 sector 文件
    out_path = os.path.join(DATA_PATH, 'sector_price_day.parquet')
    sector_old = read_parquet_safe(out_path)
    last_day = None
    if sector_old is not None and not sector_old.empty:
        try:
            last_day = pd.to_datetime(sector_old['date']).max().strftime('%Y-%m-%d') if 'date' in sector_old.columns else pd.to_datetime(sector_old.index).max().strftime('%Y-%m-%d')
        except Exception:
            last_day = None

    # 增量区间（包含 last_day 当天以便覆盖）
    fetch_start = start if last_day is None else last_day
    price_small = price_small[(price_small['date'] >= pd.to_datetime(fetch_start)) & (price_small['date'] <= pd.to_datetime(end))]
    if price_small.empty:
        return

    # 当日每风格聚合：新增 breadth_style 列
    def _agg_one_day(df_day: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for sec in TARGET_SECTORS:
            sub = df_day[df_day['sector'] == sec]
            if sub.empty:
                continue
            # 有效样本：需要 close 与 total_turnover 有效
            sub = sub.copy()
            sub = sub.replace([np.inf, -np.inf], np.nan)
            sub = sub.dropna(subset=['close', 'total_turnover'])
            if sub.empty:
                continue

            # 计算 breadth（这里用 close 与 open 近似当日收益，你可以改成使用 prev_close）
            # 只对 open 和 close 都非空的记录计算涨跌
            if 'open' in sub.columns:
                sub_ret = sub.dropna(subset=['open', 'close']).copy()
                if len(sub_ret) > 0:
                    ret = sub_ret['close'] / sub_ret['open'] - 1.0
                    up_count = (ret > 0).sum()
                    down_count = (ret < 0).sum()
                    total = len(ret)
                    if total > 0:
                        breadth_val = float((up_count - down_count) / total)
                    else:
                        breadth_val = np.nan
                else:
                    breadth_val = np.nan
            else:
                breadth_val = np.nan

            tot = float(sub['total_turnover'].sum())
            if tot <= 0:
                # 无成交额：close 用简单平均兜底
                close_val = float(sub['close'].mean())
            else:
                w = sub['total_turnover'].values.astype(float) / tot
                close_val = float(np.sum(sub['close'].values.astype(float) * w))

            volume_sum = float(sub['volume'].sum()) if 'volume' in sub.columns else np.nan
            turnover_sum = float(sub['total_turnover'].sum())
            trades_sum = float(sub['num_trades'].sum()) if 'num_trades' in sub.columns else np.nan
            cnt = int(len(sub))

            rows.append({
                'sector': sec,
                'date': df_day['date'].iloc[0],
                'close': close_val,
                'total_turnover': turnover_sum,
                'volume': volume_sum,
                'num_trades': trades_sum,
                'constituents_count': cnt,
                'breadth': breadth_val,          # 新增：风格 breadth
            })
        if not rows:
            return pd.DataFrame(columns=['sector','date','close','total_turnover','volume','num_trades','constituents_count','breadth'])
        return pd.DataFrame(rows)

    # 对每日进行聚合
    grouped = price_small.groupby('date', sort=True)
    out_parts = []
    for dt, df_day in grouped:
        out_parts.append(_agg_one_day(df_day))

    if not out_parts:
        return
    sector_new = pd.concat(out_parts, ignore_index=True)
    sector_new = sector_new.sort_values(['date','sector'])

    # 合并旧数据并去重（保留新行覆盖）
    if sector_old is None or sector_old.empty:
        sector_all = sector_new
    else:
        # 统一列顺序（加上 breadth）
        base_cols = ['sector','date','close','total_turnover','volume','num_trades','constituents_count','breadth']
        for c in base_cols:
            if c not in sector_old.columns:
                sector_old[c] = np.nan
        sector_old = sector_old[base_cols].copy()

        sector_all = pd.concat([sector_old, sector_new], axis=0, ignore_index=True)
        sector_all['date'] = pd.to_datetime(sector_all['date'])
        sector_all = sector_all.drop_duplicates(subset=['sector','date'], keep='last').sort_values(['date','sector'])

    write_parquet(sector_all, out_path)
    print(f'[DATA] 已更新风格日行情：{out_path}，总行数={len(sector_all)}，新增/覆盖={len(sector_new)}')


# =========================
# 主流程
# =========================
def main():
    # 1) 交易日 & 周
    print('更新交易日与交易周...')
    update_trading_calendar(start='2001-01-01', end=TRADING_DAY_END)

    # 2) 股票信息（JQ 导出覆盖）
    print('更新股票信息（需先导出 stock_list_jq.csv）...')
    update_stock_info_from_jq_export()

    # 3) 股票日行情（前复权，MultiIndex）
    print('更新股票日行情...')
    update_stock_price_day(start='2010-01-01', end=TODAY_STR)

    # 3.5) 股票基本面日行情（前复权，MultiIndex）
    print('更新股票基本面日行情...')
    update_stock_fundamental_day(start='2010-01-01', end=TODAY_STR, factor_list = FACTOR_LIST)

    # 4) 停牌（宽表）
    print('更新停牌信息...')
    update_suspended(start='2010-01-01', end=TODAY_STR)

    # 5) ST（宽表）
    print('更新 ST 信息...')
    update_is_st(start='2010-01-01', end=TODAY_STR)

    # 6) 指数成分（快照覆盖）
    print('更新指数成分...')
    update_index_components()

    # 7) 行业与风格（快照覆盖 + 映射）
    print('更新行业与风格...')
    update_industry_and_style()

    # 8) 指数日行情（MultiIndex）
    print('更新指数日行情...')
    update_index_price_day(start='2010-01-01', end=TODAY_STR)

    # 9) 新增：风格日行情（增量聚合）
    print('更新风格日行情（sector_price_day）...')
    update_sector_price_day(start='2010-01-01', end=TODAY_STR)

    # 10) 指数净值序列
    print('更新指数净值序列...')
    indexs = pd.read_parquet('data/raw/index_price_day.parquet').reset_index()
    # 确保 df300 是独立的 DataFrame（避免视图问题）
    for id in ['000300.XSHG','000852.XSHG','000905.XSHG']:
        df = indexs[indexs['order_book_id'] == id].copy()
        df['ret'] = (df['close'] - df['prev_close']) / df['prev_close']
        df['nav'] = (1 + df['ret']).cumprod()

        selected_columns = df[['date', 'ret', 'nav']].rename(columns={'ret': 'ret_total'})
        selected_columns.to_csv(f'backtest_rolling/others/{id[3:-5]}_index_nav.csv', index=False)

    # # 11) 更新黄金ETF
    # print('更新黄金ETF')
    # gold_etf = ak.fund_etf_hist_em(symbol="159934", period="daily", start_date="20100101", end_date=date.today().strftime("%Y%m%d"), adjust="qfq")
    # gold_etf = gold_etf[['日期','开盘','收盘']]
    # gold_etf.columns = ['date','open','close']
    # gold_etf.to_parquet('data/raw/gold_etf_day.parquet')


if __name__ == '__main__':
    main()