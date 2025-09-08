# coding: utf-8
"""
诊断：为什么某周筛选后为空？（自动对齐 H5 组日期与口径起点）
- 允许输入任意自然日；脚本会映射到该周“有效周五”与 H5 组
- 分步统计每条过滤规则的剔除数量与样例，定位元凶
"""

from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
import h5py

from backtest_config import BT_CFG
from config import CFG
from utils import load_calendar, weekly_fridays
from train_utils import make_filter_fn, load_stock_info, load_flag_table


def ensure_price_multi(price_df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(price_df.index, pd.MultiIndex):
        return price_df.sort_index()
    if {"order_book_id", "date"}.issubset(price_df.columns):
        price_df["date"] = pd.to_datetime(price_df["date"])
        return price_df.set_index(["order_book_id", "date"]).sort_index()
    raise ValueError("price_day_file 需包含 MultiIndex (order_book_id, date) 或列 ['order_book_id','date'].")


def list_h5_groups(h5: h5py.File) -> pd.DataFrame:
    rows = []
    for k in h5.keys():
        if not k.startswith("date_"):
            continue
        d = h5[k].attrs.get("date", None)
        if d is None:
            continue
        if isinstance(d, bytes):
            d = d.decode("utf-8")
        rows.append({"group": k, "date": pd.to_datetime(d)})
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


def build_date_maps(h5: h5py.File, cal: pd.DatetimeIndex, price_df: pd.DataFrame):
    # 1) H5 的 date->group
    df = list_h5_groups(h5)
    date2grp = {row["date"]: row["group"] for _, row in df.iterrows()}
    h5_dates = pd.DatetimeIndex(sorted(date2grp.keys()))

    # 2) 周五序列
    fridays_all = weekly_fridays(cal)

    # 3) 计算每个周五的周收益“起点交易日 d0”
    close_pivot = price_df["close"].unstack(0).sort_index()
    avail = close_pivot.index

    friday_to_start = {}
    for i in range(len(fridays_all) - 1):
        f0 = fridays_all[i]
        f1 = fridays_all[i + 1]
        i0 = avail[avail <= f0]
        i1 = avail[avail <= f1]
        if len(i0) == 0 or len(i1) == 0:
            continue
        d0 = i0[-1]; d1 = i1[-1]
        if d0 >= d1:
            continue
        friday_to_start[f0] = d0

    return date2grp, h5_dates, fridays_all, friday_to_start


def map_input_date_to_h5_group(input_date: pd.Timestamp,
                               h5_dates: pd.DatetimeIndex,
                               fridays_all: pd.DatetimeIndex) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    输入任意自然日，返回：
    - f0: 不晚于 input_date 的最近一个“周五”（基于周五序列）
    - g_date: 不晚于 f0 的最近一个 H5 组日期（用于从 H5 读取该周组）
    若映射失败，抛异常。
    """
    # 就近找 <= input_date 的周五
    fr_ok = fridays_all[fridays_all <= input_date]
    if len(fr_ok) == 0:
        raise RuntimeError(f"找不到 <= {input_date.date()} 的周五")
    f0 = fr_ok[-1]

    # 再找 <= f0 的 H5 组日期
    h5_ok = h5_dates[h5_dates <= f0]
    if len(h5_ok) == 0:
        raise RuntimeError(f"找不到 <= {f0.date()} 的 H5 组日期（H5 覆盖可能从更晚时间开始）")
    g_date = h5_ok[-1]
    return f0, g_date


def apply_board_switch_mask(codes: np.ndarray) -> np.ndarray:
    keep = np.ones(codes.shape[0], dtype=bool)
    if not CFG.include_star_market:
        keep &= ~(np.char.startswith(codes, "688") | np.char.startswith(codes, "689"))
    if not CFG.include_chinext:
        keep &= ~(np.char.startswith(codes, "300") | np.char.startswith(codes, "301"))
    if not CFG.include_bse:
        has_xbei = np.char.find(codes, ".XBEI") >= 0
        has_xbse = np.char.find(codes, ".XBSE") >= 0
        keep &= ~(has_xbei | has_xbse)
    if not CFG.include_neeq:
        is_neeq = np.zeros_like(keep)
        for suf in [".XNE", ".XNEE", ".XNEQ", ".XNEX"]:
            is_neeq |= (np.char.find(codes, suf) >= 0)
        keep &= ~is_neeq
    return keep


def diagnose_for_date(input_date_str: str):
    input_date = pd.to_datetime(input_date_str)
    print("\n================ 诊断输入日期:", input_date.date(), "================")

    # — 数据载入 —
    with h5py.File(BT_CFG.feat_file, "r") as h5:
        df_groups = list_h5_groups(h5)
        price_df = pd.read_parquet(BT_CFG.price_day_file)
        price_df = ensure_price_multi(price_df)
        cal = load_calendar(BT_CFG.trading_day_file)

        date2grp, h5_dates, fridays_all, friday_to_start = build_date_maps(h5, cal, price_df)

        # 将输入日期映射到 f0（周五）与 g_date（H5 组日期）
        try:
            f0, g_date = map_input_date_to_h5_group(input_date, h5_dates, fridays_all)
        except Exception as e:
            print("映射失败：", e)
            return

        print(f"映射到周五 f0={f0.date()}；对应 H5 组日期 g_date={g_date.date()}")
        if f0 not in friday_to_start:
            print("找不到该周的收益起点 d0（friday_to_start 缺失），跳过。")
            return
        d0 = friday_to_start[f0]
        print("口径起点交易日 d0:", d0.date())

        # 取当周 H5 组
        gk = date2grp[g_date]
        g = h5[gk]
        stocks_all = g["stocks"][:].astype(str)
        codes = np.asarray(stocks_all, dtype=str)
        print("H5 候选股票数:", len(codes))

    # — 构造 filter_fn，与回测参数对齐 —
    CFG.enable_filters = bool(BT_CFG.enable_filters)
    CFG.ipo_cut_days = int(BT_CFG.ipo_cut_days)
    CFG.suspended_exclude = bool(BT_CFG.suspended_exclude)
    CFG.st_exclude = bool(BT_CFG.st_exclude)
    CFG.min_daily_turnover = float(BT_CFG.min_daily_turnover)
    CFG.allow_missing_info = bool(BT_CFG.allow_missing_info)
    CFG.include_star_market = bool(getattr(BT_CFG, "include_star_market", True))
    CFG.include_chinext = bool(getattr(BT_CFG, "include_chinext", True))
    CFG.include_bse = bool(getattr(BT_CFG, "include_bse", True))
    CFG.include_neeq = bool(getattr(BT_CFG, "include_neeq", True))

    stock_info_df = load_stock_info(BT_CFG.stock_info_file)
    susp_df = load_flag_table(BT_CFG.is_suspended_file)
    st_df = load_flag_table(BT_CFG.is_st_file)
    filter_fn = make_filter_fn(price_df, stock_info_df, susp_df, st_df)

    # — 分步筛选统计 —
    stats = []
    # a) 板块
    keep_mask_board = apply_board_switch_mask(codes)
    after_board = codes[keep_mask_board]
    stats.append(("board_switch", len(codes) - len(after_board), after_board))

    # b) 停牌
    after_b = after_board
    if CFG.suspended_exclude and (susp_df is not None) and (d0 in susp_df.index):
        row = susp_df.loc[d0]
        keep = [s for s in after_b if (s in row.index and int(row.get(s, 0)) == 0)]
        stats.append(("suspended", len(after_b) - len(keep), np.array(keep, dtype=str)))
        after_b = np.array(keep, dtype=str)
    else:
        stats.append(("suspended", 0, after_b))

    # c) ST
    after_c = after_b
    if CFG.st_exclude and (st_df is not None) and (d0 in st_df.index):
        row = st_df.loc[d0]
        keep = [s for s in after_c if (s in row.index and int(row.get(s, 0)) == 0)]
        stats.append(("st_flag", len(after_c) - len(keep), np.array(keep, dtype=str)))
        after_c = np.array(keep, dtype=str)
    else:
        stats.append(("st_flag", 0, after_c))

    # d) 成交额
    after_d = after_c
    to_removed = 0
    if CFG.min_daily_turnover and CFG.min_daily_turnover > 0:
        try:
            slice_dt = price_df.xs(d0, level=1, drop_level=False)
            to_col = None
            for c in ["total_turnover", "turnover", "amount"]:
                if c in slice_dt.columns:
                    to_col = c; break
            if to_col is None:
                print("未找到成交额列 total_turnover/turnover/amount，跳过该步。")
            else:
                idx = [(s, d0) for s in after_d]
                sub = slice_dt.loc[idx]
                ok = sub[sub[to_col] >= CFG.min_daily_turnover].index.get_level_values(0).unique().astype(str).tolist()
                to_removed = len(after_d) - len(ok)
                after_d = np.array(ok, dtype=str)
        except KeyError:
            present = slice_dt.index.get_level_values(0).astype(str).unique().tolist()
            ok = [s for s in after_d if s in present]
            to_removed = len(after_d) - len(ok)
            after_d = np.array(ok, dtype=str)
        except Exception as e:
            print("[WARN] 成交额阈值筛选异常：", e)
    stats.append(("turnover_min", to_removed, after_d))

    # e) IPO/退市/缺信息（借助 filter_fn 逐只）
    final_pool = []
    removed_ipo = removed_delist = removed_missing = removed_other = 0
    for s in after_d:
        ok = False
        try:
            ok = bool(filter_fn(d0, s))
        except Exception:
            ok = False
        if ok:
            final_pool.append(s)
        else:
            # 粗略分类
            if s not in stock_info_df.index:
                removed_missing += 1
            else:
                row = stock_info_df.loc[s]
                ipo = row.get("ipo_date", pd.NaT)
                delist = row.get("delist_date", pd.NaT)
                if pd.notna(delist) and pd.Timestamp(d0) >= pd.Timestamp(delist):
                    removed_delist += 1
                elif pd.notna(ipo) and (pd.Timestamp(d0) - pd.Timestamp(ipo)).days < CFG.ipo_cut_days:
                    removed_ipo += 1
                else:
                    removed_other += 1
    final_pool = np.array(final_pool, dtype=str)

    # 输出
    print("\n—— 分步统计 ——")
    cur = codes
    for name, removed, after in stats:
        print(f"{name:<14} 剔除 {removed:>5} / 前 {len(cur):>5} -> 后 {len(after):>5}")
        cur = after

    print(f"\nIPO不足天数剔除: {removed_ipo}")
    print(f"退市口径剔除    : {removed_delist}")
    print(f"缺少基础信息剔除: {removed_missing}")
    print(f"其他规则剔除    : {removed_other}")
    print(f"\n最终剩余（估计）：{len(final_pool)} 支")

    if len(final_pool) == 0:
        # 找最有可能的“元凶”
        culprit = None
        cur = codes
        for (name, removed, after) in stats:
            if len(after) == 0:
                culprit = name; break
            cur = after
        if culprit is None:
            # 前四步后非空，则看 e 步里哪个计数最大
            cul_counts = {
                "ipo_days": removed_ipo,
                "delist": removed_delist,
                "missing_info": removed_missing,
                "other": removed_other
            }
            culprit = max(cul_counts, key=lambda k: cul_counts[k])
        print(f"[提示] 本周股票池为空的最可能原因：{culprit}")

    # 打印每环节剔除样例
    def show_examples(step_name: str, before_arr: np.ndarray, after_arr: np.ndarray, max_show=10):
        removed = list(set(before_arr) - set(after_arr))
        if removed:
            print(f"\n{step_name} 剔除样例（最多{max_show}只）：")
            for s in removed[:max_show]:
                print("  -", s)

    arr0 = codes
    arr1 = stats[0][2]
    arr2 = stats[1][2]
    arr3 = stats[2][2]
    arr4 = stats[3][2]
    show_examples("board_switch", arr0, arr1)
    show_examples("suspended",   arr1, arr2)
    show_examples("st_flag",      arr2, arr3)
    show_examples("turnover_min", arr3, arr4)

    print("\n—— 参数回显 ——")
    print(f"enable_filters={CFG.enable_filters}, ipo_cut_days={CFG.ipo_cut_days}, "
          f"suspended_exclude={CFG.suspended_exclude}, st_exclude={CFG.st_exclude}, "
          f"min_daily_turnover={CFG.min_daily_turnover}, allow_missing_info={CFG.allow_missing_info}")
    print(f"include_star_market={CFG.include_star_market}, include_chinext={CFG.include_chinext}, "
          f"include_bse={CFG.include_bse}, include_neeq={CFG.include_neeq}")


def main():
    # 把这里替换成你要对比的两个日期（可以是自然日，不要求恰好是 H5 组日期）
    target_dates = ["2025-08-29", "2025-09-05"]
    for ds in target_dates:
        diagnose_for_date(ds)


if __name__ == "__main__":
    main()