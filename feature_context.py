# coding: utf-8
"""
增量生成 + 预处理（无未来信息）的市场/风格上下文特征：context_features.parquet
- 生成缺失周五的“原始上下文特征”（指数与风格统计），与旧文件合并去重；
- 使用固定滚动窗口做历史标准化（仅用历史，不引入未来）；
- 在首个达到参考样本数的历史窗口，先“方差筛选”（剔除低方差列），再“相关性阈值剔除”，得到稳定列集合；
- 将列集合保存到 ctx_selected_cols.txt，后续增量沿用；
- 输出 processed/context_features.parquet 供训练/回测读取。

保证无未来信息：
- 标准化：对每个日期 t，仅使用 t 之前（含 t）的滚动窗口统计；
- 列筛选：仅用首个历史窗口（长度>=CORR_REF_MIN）的数据一次性确定列集合，之后固定不变。
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List

from config import CFG
from utils import load_calendar, weekly_fridays

# ====================== 可调参数（预处理） ======================
USE_EXPANDING = False     # False：使用固定滚动窗口；True：扩张窗口
ROLLING_WIN   = 52       # 滚动窗口长度（单位=周），如 104≈2年
CORR_REF_MIN  = 26       # 首个参考历史窗口的样本数下限（>=该样本数时执行列筛选）
VAR_THRESH    = 1e-8      # 方差筛选阈值：参考窗口内方差 < 该值的列将被剔除（避免常量/近似常量列）
CORR_THRESH   = 0.9      # 绝对相关性阈值：|corr| >= 阈值 的特征对进行二选一保留
Z_EPS         = 1e-6

# 列集合落盘（固定列集合，便于增量复用）
COLS_FILE     = CFG.processed_dir / "ctx_selected_cols.txt"

# ====================== 原始上下文构造参数 ======================
EPS = 1e-12
SECTORS = ['周期风格', '成长风格', '消费风格', '稳定风格', '金融风格']
INDEXS  = ['000300.XSHG', '000905.XSHG', '000852.XSHG']

def _mean(s: pd.Series, n: int): return s.rolling(n, min_periods=1).mean()
def _std (s: pd.Series, n: int): return s.rolling(n, min_periods=1).std()

# ====================== 数据准备（与原版一致） ======================
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
                            target_fridays: pd.DatetimeIndex) -> pd.DataFrame:
    """仅对给定的 target_fridays 计算上下文特征（原始，未标准化/筛选）"""
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

# ====================== 预处理（方差筛选 + 相关性剔除 + 标准化） ======================
def select_columns_with_variance_and_corr(df: pd.DataFrame,
                                          ref_min: int,
                                          var_thresh: float,
                                          corr_thresh: float) -> List[str]:
    """
    仅在首个历史窗口（长度>=ref_min）上确定列集合：
      1) 方差筛选：剔除参考窗口内方差 < var_thresh 的列（常量/近似常量）；
      2) 相关性阈值剔除：|corr| >= corr_thresh 时保留“缺失率更低、历史方差更大”的列；
      3) 输出按原列顺序保留的列集合。
    若已有落盘列集合（ctx_selected_cols.txt），则直接加载并与当前列求交集。
    """
    # 优先复用既有列集合
    if COLS_FILE.exists():
        cols = [line.strip() for line in COLS_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
        cols = [c for c in cols if c in df.columns]
        if len(cols) == 0:
            raise RuntimeError("ctx_selected_cols.txt 为空或与当前数据列不匹配，请删除该文件以重新选择。")
        print(f"[Context] 载入既有列集合：{len(cols)} 列")
        return cols

    # 若样本不足，暂不筛选
    if df.shape[0] < ref_min:
        print(f"[Context][警告] 行数 {df.shape[0]} < 参考最小样本 {ref_min}，暂不做列筛选，保留全部列。")
        cols_final = df.columns.tolist()
    else:
        df_ref = df.iloc[:ref_min].copy()

        # 1) 方差筛选
        var_series = df_ref.var(axis=0, ddof=1).fillna(0.0)
        mask_var = var_series >= var_thresh
        if mask_var.sum() == 0:
            print(f"[Context][警告] 方差筛选后无特征保留（阈值={var_thresh}），回退保留全部列。")
            cols_after_var = df_ref.columns.tolist()
        else:
            cols_after_var = var_series.index[mask_var].tolist()
        print(f"[Context] 方差筛选：{df_ref.shape[1]} -> {len(cols_after_var)}（阈值={var_thresh}）")

        # 为相关性计算准备数据（仅保留方差通过的列）
        df_ref2 = df_ref[cols_after_var]

        # 质量指标（缺失率/方差，用于相关性冲突时的保留规则）
        miss_rate = df_ref2.isna().mean(axis=0)            # 缺失率低优先
        var_val   = df_ref2.var(axis=0, ddof=1).fillna(0)  # 方差大优先

        # 仅用于相关性计算的填充（不回写）
        ref_filled = df_ref2.fillna(df_ref2.median()).fillna(0.0)
        corr = ref_filled.corr().abs().values
        cols_all = df_ref2.columns.tolist()
        keep = np.ones(len(cols_all), dtype=bool)

        # 2) 相关性阈值剔除（贪心）
        for i in range(len(cols_all)):
            if not keep[i]:
                continue
            for j in range(i+1, len(cols_all)):
                if not keep[j]:
                    continue
                cij = corr[i, j]
                if np.isnan(cij):
                    continue
                if cij >= corr_thresh:
                    ai = (float(miss_rate[cols_all[i]]), float(-var_val[cols_all[i]]))
                    aj = (float(miss_rate[cols_all[j]]), float(-var_val[cols_all[j]]))
                    if ai <= aj:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break

        cols_final = [c for c, k in zip(cols_all, keep) if k]
        print(f"[Context] 相关性筛选：{len(cols_all)} -> {len(cols_final)}（阈值={corr_thresh}，参考样本={ref_min}）")

    # 落盘列集合
    COLS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(COLS_FILE, "w", encoding="utf-8") as f:
        for c in cols_final:
            f.write(c + "\n")
    print(f"[Context] 已保存列集合到：{COLS_FILE}")
    return cols_final

def historical_standardize(df: pd.DataFrame,
                           use_expanding: bool,
                           rolling_win: int) -> pd.DataFrame:
    """
    历史标准化：对每一列做 (x - mean_hist) / std_hist（不引入未来信息）
    - expanding: mean_t=mean(df[:t]), std_t=std(df[:t])
    - rolling : 固定窗口长度的滚动统计
    """
    df = df.astype(float)
    if use_expanding:
        mean_exp = df.expanding(min_periods=2).mean()
        std_exp  = df.expanding(min_periods=2).std(ddof=1)
        out = (df - mean_exp) / (std_exp + Z_EPS)
    else:
        mean_roll = df.rolling(window=rolling_win, min_periods=2).mean()
        std_roll  = df.rolling(window=rolling_win, min_periods=2).std(ddof=1)
        out = (df - mean_roll) / (std_roll + Z_EPS)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out

# ====================== 主流程（合并：生成 + 预处理） ======================
def main():
    out_path = CFG.processed_dir / "context_features.parquet"

    # 交易周五全集
    cal = load_calendar(CFG.trading_day_file)
    fridays_all = weekly_fridays(cal)
    fridays_all = fridays_all[(fridays_all >= pd.Timestamp(CFG.start_date)) &
                              (fridays_all <= pd.Timestamp(CFG.end_date))]

    # 已有 context（可能为历史输出；合并后统一再预处理）
    if out_path.exists():
        ctx_old = pd.read_parquet(out_path)
        if not isinstance(ctx_old.index, pd.DatetimeIndex):
            raise AssertionError("context_features 的索引必须是 DatetimeIndex")
        existing_dates = pd.DatetimeIndex(sorted(set(ctx_old.index)))
        last_existing = existing_dates.max() if len(existing_dates) > 0 else None
        print(f"[Context增量] 现有行数={len(ctx_old)}, 覆盖周五={len(existing_dates)}")
    else:
        ctx_old = None
        existing_dates = pd.DatetimeIndex([])
        last_existing = None

    # 需要增量的周五（缺失 ∪ 最后一个已存在周五用于覆盖）
    missing = [d for d in fridays_all if d not in set(existing_dates)]
    if last_existing is not None and last_existing not in missing:
        missing.append(last_existing)
    missing = pd.DatetimeIndex(sorted(set(missing)))

    if len(missing) == 0:
        print("[Context增量] 无需更新，仍会基于现有文件执行预处理检查。")
        ctx_all_raw = ctx_old.copy()
    else:
        print(f"[Context增量] 待计算周五数量={len(missing)} "
              f"(范围: {missing.min().date()} ~ {missing.max().date()})")

        # 载入原始数据
        index_df  = pd.read_parquet(CFG.index_day_file)
        sector_df = pd.read_parquet(CFG.style_day_file)

        # 仅对目标周五计算原始特征
        ctx_new_raw = build_context_for_dates(index_df, sector_df, missing)

        # 合并（新结果覆盖旧）
        if ctx_old is None or ctx_old.empty:
            ctx_all_raw = ctx_new_raw
        else:
            ctx_all_raw = pd.concat([ctx_old, ctx_new_raw], axis=0)
            ctx_all_raw = ctx_all_raw[~ctx_all_raw.index.duplicated(keep='last')].sort_index()

    if ctx_all_raw is None or ctx_all_raw.empty:
        print("[Context][警告] 上下文原始数据为空，跳过保存。")
        return

    # 首窗：方差筛选 + 相关性剔除（或加载已有列集合）
    cols_selected = select_columns_with_variance_and_corr(
        ctx_all_raw, ref_min=CORR_REF_MIN, var_thresh=VAR_THRESH, corr_thresh=CORR_THRESH
    )
    ctx_all_raw = ctx_all_raw.reindex(columns=cols_selected)

    # 历史标准化（滚动/扩张，均不使用未来信息）
    ctx_std = historical_standardize(ctx_all_raw, use_expanding=USE_EXPANDING, rolling_win=ROLLING_WIN)

    # 保存最终上下文
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ctx_std.to_parquet(out_path)
    print(f"[Context] 保存完成：{out_path}，总行数={len(ctx_std)}，列数={ctx_std.shape[1]} "
          f"(expanding={USE_EXPANDING}, rolling_win={ROLLING_WIN if not USE_EXPANDING else 'N/A'}, "
          f"var_thresh={VAR_THRESH}, corr_thresh={CORR_THRESH})")

if __name__ == "__main__":
    main()