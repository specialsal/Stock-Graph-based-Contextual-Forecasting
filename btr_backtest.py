# coding: utf-8
"""
btr_backtest.py
读取 positions_{run_name}_{weight_mode}.csv + 原始日线，构建周度组合净值：
- 调仓周：按 O2C 口径并在价格层计入单股买卖成本/滑点；
- 非调仓周：按 C2C 口径且不计任何交易成本（真实“持有不过度交易”）。
- 写 backtest_rolling/{run_name}/nav_{run_name}.csv（不带 weight_mode 后缀）

调仓周识别（基于持仓变化）：
- 第一个有持仓的周视为调仓周；
- 后续周：若本周与上周的股票集合或任意权重有显著变化（阈值 eps）则为调仓周，否则为非调仓周。
"""

import math
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

from backtest_rolling_config import BT_ROLL_CFG
from utils import load_calendar, weekly_fridays


def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def _cost_terms_from_cfg():
    slp = float(BT_ROLL_CFG.slippage_bps) * 1e-4
    buy_fee = float(BT_ROLL_CFG.buy_fee_bps) * 1e-4
    sell_fee = float(BT_ROLL_CFG.sell_fee_bps) * 1e-4
    buy_cost = slp + buy_fee
    sell_cost = slp + sell_fee
    return buy_cost, sell_cost


def _build_pivots(price_df: pd.DataFrame):
    """构造 open/high/low/close 的宽表（index=date, columns=stock）"""
    close_pivot = price_df["close"].unstack(0).sort_index()
    open_pivot  = price_df["open"].unstack(0).sort_index()
    high_pivot  = price_df["high"].unstack(0).sort_index()
    low_pivot   = price_df["low"].unstack(0).sort_index()
    return open_pivot, high_pivot, low_pivot, close_pivot


def _align_last(avail_idx: pd.DatetimeIndex, target: pd.Timestamp) -> Optional[pd.Timestamp]:
    """对齐至不晚于 target 的最后一个交易日"""
    ok = avail_idx[avail_idx <= target]
    return ok[-1] if len(ok) > 0 else None


def _week_ret_o2c_with_cost(open_pv: pd.DataFrame, close_pv: pd.DataFrame,
                            f0: pd.Timestamp, f1: pd.Timestamp,
                            buy_cost: float, sell_cost: float) -> Optional[pd.Series]:
    """
    调仓周：O2C 含成本。返回 Series(index=stock)。
    """
    avail = close_pv.index
    d0 = _align_last(avail, f0)
    d1 = _align_last(avail, f1)
    if d0 is None or d1 is None:
        return None

    # 起点为 d0 的下一交易日
    pos = avail.get_indexer([d0])[0]
    if pos < 0 or pos + 1 >= len(avail):
        return None
    d_start = avail[pos + 1]
    if d_start > d1:
        return None

    try:
        O0 = open_pv.loc[d_start]
        C1 = close_pv.loc[d1]
    except KeyError:
        return None

    O0_eff = O0 * (1.0 + buy_cost)
    C1_eff = C1 * (1.0 - sell_cost)
    ret = (C1_eff / O0_eff - 1.0).replace([np.inf, -np.inf], np.nan)
    return ret


def _week_ret_c2c_nocost(close_pv: pd.DataFrame,
                         f_prev: pd.Timestamp, f_cur: pd.Timestamp) -> Optional[pd.Series]:
    """
    非调仓周：C2C 不计成本。返回 Series(index=stock)。
    """
    avail = close_pv.index
    d0 = _align_last(avail, f_prev)
    d1 = _align_last(avail, f_cur)
    if d0 is None or d1 is None or d1 <= d0:
        return None
    try:
        C0 = close_pv.loc[d0]
        C1 = close_pv.loc[d1]
    except KeyError:
        return None
    ret = (C1 / C0 - 1.0).replace([np.inf, -np.inf], np.nan)
    return ret


def _is_rebalance_week(w_prev: Optional[pd.Series],
                       w_curr: pd.Series,
                       eps: float = 1e-9) -> bool:
    """
    基于“持仓变化”识别调仓周：
    - 第一个有持仓周或上周无权重视为调仓周；
    - 股票集合变化或任意权重变化超过阈值则为调仓周；否则非调仓周。
    参数：
      w_prev/w_curr: Series(index=stock, value=weight)，已归一化或原始权重均可
    """
    if w_curr is None or w_curr.empty:
        return False
    if (w_prev is None) or (w_prev.empty):
        return True  # 第一周有持仓 => 调仓

    # 集合差异
    set_prev = set(w_prev.index)
    set_curr = set(w_curr.index)
    if set_prev != set_curr:
        return True

    # 权重差异（对齐比较）
    w_prev2 = w_prev.reindex(w_curr.index).fillna(0.0).values
    w_curr2 = w_curr.fillna(0.0).values
    diff = np.abs(w_curr2 - w_prev2).max() if len(w_curr2) > 0 else 0.0
    return bool(diff > eps)


def _pivot_weights_for_week(positions: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    """
    从明细中抽取当周权重向量 Series(index=stock, value=weight)；剔除非有限值与零权重。
    """
    g = positions[positions["date"] == date]
    if g.empty:
        return pd.Series(dtype=float)
    s = pd.Series(g["weight"].values.astype(float), index=g["stock"].astype(str).values)
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    s = s[s != 0.0]
    return s


def main():
    cfg = BT_ROLL_CFG
    out_dir = cfg.backtest_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    weight_mode = str(getattr(cfg, "weight_mode", "equal")).lower()
    # 输入持仓：positions_{run_name}_{weight_mode}.csv
    pos_path = out_dir / f"positions_{cfg.run_name_out}_{weight_mode}.csv"
    if not pos_path.exists():
        raise FileNotFoundError(f"未找到持仓文件：{pos_path}。请先运行 optimize_position.py 生成持仓。")

    positions = pd.read_csv(pos_path)
    if positions.empty:
        raise RuntimeError(f"持仓文件为空：{pos_path}")
    if not {"date", "stock", "weight"}.issubset(set(positions.columns)):
        raise RuntimeError("持仓文件缺少必要列：需要包含 date, stock, weight")

    # 规范
    positions["date"] = pd.to_datetime(positions["date"])
    positions = positions.sort_values(["date", "stock"])
    # 限定回测区间（防御）
    positions = positions[(positions["date"] >= pd.Timestamp(cfg.bt_start_date)) &
                          (positions["date"] <= pd.Timestamp(cfg.bt_end_date))]

    # 价格日线
    price_df = pd.read_parquet(cfg.price_day_file)
    if not isinstance(price_df.index, pd.MultiIndex):
        price_df = price_df.set_index(["order_book_id", "date"]).sort_index()
    need_cols = ["open", "high", "low", "close"]
    lack = [c for c in need_cols if c not in price_df.columns]
    if lack:
        raise RuntimeError(f"price_day_file 缺少必要列：{lack}")

    open_pivot, high_pivot, low_pivot, close_pivot = _build_pivots(price_df)

    # 周锚
    cal = load_calendar(cfg.trading_day_file)
    fridays_all = weekly_fridays(cal)

    # 仅保留持仓出现的周与可用的周五锚点的交集，且在回测区间内
    weeks = sorted(positions["date"].unique())
    weeks = [pd.Timestamp(d) for d in weeks if (d in fridays_all)]
    weeks = [d for d in weeks if (d >= pd.Timestamp(cfg.bt_start_date)) and (d <= pd.Timestamp(cfg.bt_end_date))]
    if len(weeks) < 1:
        print("[BTR-BT] 有效周度样本不足，退出。")
        return

    buy_cost, sell_cost = _cost_terms_from_cfg()

    # 遍历周，先判定调仓与否，再据此选择 O2C(含成本) 或 C2C(无成本) 的单股收益
    recs = []
    stock_recs = []

    prev_weights_vec: Optional[pd.Series] = None  # index=stock, value=weight
    for i, d_cur in enumerate(weeks):
        # 当周权重向量
        w_cur = _pivot_weights_for_week(positions, d_cur)

        # 判定是否调仓周
        is_rebal = _is_rebalance_week(prev_weights_vec, w_cur, eps=1e-9)

        # 找下一周锚点（用于收益窗口右端）
        if i < len(weeks) - 1:
            d_next = weeks[i + 1]
        else:
            # 最后一周：用 fridays_all 中 d_cur 之后的下一个周五作为右端（若存在）
            idx = fridays_all.get_indexer([d_cur])[0]
            d_next = fridays_all[idx + 1] if (idx >= 0 and idx + 1 < len(fridays_all)) else None

        if d_next is None:
            # 无法构造下一周收益，跳过
            prev_weights_vec = w_cur
            continue

        # 选择收益口径
        if is_rebal:
            # 调仓：O2C 含成本
            ret_s = _week_ret_o2c_with_cost(open_pivot, close_pivot, d_cur, d_next, buy_cost, sell_cost)
        else:
            # 非调仓：C2C 无成本（从上一周末收盘到本周末收盘）
            ret_s = _week_ret_c2c_nocost(close_pivot, d_cur, d_next)

        if ret_s is None or ret_s.empty or w_cur is None or w_cur.empty:
            # 无有效收益或当周空仓
            recs.append({"date": d_cur, "ret_long": 0.0, "ret_short": 0.0, "ret_total": 0.0, "n_long": 0})
            prev_weights_vec = w_cur
            continue

        # 对齐到持仓股票
        sub = w_cur.to_frame(name="weight").merge(ret_s.rename("ret"), left_index=True, right_index=True, how="inner")
        sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=["weight", "ret"])
        if sub.empty:
            recs.append({"date": d_cur, "ret_long": 0.0, "ret_short": 0.0, "ret_total": 0.0, "n_long": 0})
            prev_weights_vec = w_cur
            continue

        # 周收益（多头）：权重 × 单股周收益
        sub["contribution"] = sub["weight"].astype(float) * sub["ret"].astype(float)

        # 记录股票明细
        for s, r in sub.iterrows():
            stock_recs.append({
                "date": d_cur,
                "stock": str(s),
                "weight": float(r["weight"]),
                "ret": float(r["ret"]),
                "contribution": float(r["contribution"])
            })

        long_ret = float(sub["contribution"].sum())
        total = float(cfg.long_weight) * long_ret  # 仅多头
        recs.append({
            "date": d_cur,
            "ret_long": long_ret,
            "ret_short": 0.0,
            "ret_total": total,
            "n_long": int(sub.shape[0])
        })

        prev_weights_vec = w_cur

    # 处理净值数据
    if not recs:
        print("[BTR-BT] 未能生成回测净值（可能所有周都空仓或无法对齐收益）。")
        return
    nav_df = pd.DataFrame(recs).set_index("date").sort_index()
    nav_df["nav"] = (1.0 + nav_df["ret_total"]).cumprod()

    # 股票收益明细
    stock_df = pd.DataFrame(stock_recs)
    if not stock_df.empty:
        stock_df["date"] = pd.to_datetime(stock_df["date"])
        stock_df = stock_df.sort_values(by=["date", "stock"]).reset_index(drop=True)

    # 输出
    out_nav_csv = out_dir / f"nav_{cfg.run_name_out}.csv"
    out_stock_csv = out_dir / f"stock_returns_{cfg.run_name_out}.csv"
    ensure_dir(out_nav_csv)
    cols = ["ret_long", "ret_short", "ret_total", "nav", "n_long"]
    nav_df[cols].to_csv(out_nav_csv, float_format="%.8f")
    if not stock_df.empty:
        stock_df.to_csv(out_stock_csv, index=False, float_format="%.8f")
    print(f"[BTR-BT] 已保存净值序列：{out_nav_csv}")
    if not stock_df.empty:
        print(f"[BTR-BT] 已保存股票收益明细：{out_stock_csv}")


if __name__ == "__main__":
    main()