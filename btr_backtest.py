# coding: utf-8
"""
btr_backtest.py（日度版，支持一次性止盈/止损 + 交易记录输出，且同一信号周期内不重复买回已止损/止盈股票）
读取日度持仓 positions_{run_name_out}_{weight_mode}.csv + 原始日线，生成“日度净值”与成交记录：
- 新增：stock_trades_{run_name_out}.csv（列：date,stock,action,entry_price,exec_price）

成交事件定义：
- BUY: 当日目标从无到有，按开盘买入，exec_price=O*(1+买入成本)，entry_price=当日开盘裸价O
- SELL: 当日目标从有到无，未触发止盈/止损，按开盘卖出，exec_price=O*(1-卖出成本)，entry_price=该轮首次入场的开盘裸价
- TP: 止盈触发清仓，exec_price=entry_price*(1+tp_ratio)*(1-卖出成本)
- SL: 止损触发清仓，exec_price=entry_price*(1-sl_ratio)*(1-卖出成本)

收益口径保持原先：
- 新增：O(今, 含买入成本)→C(今)
- 存续：C(昨)→C(今)
- 减仓/清仓：C(昨)→O(今, 含卖出成本)
- 止盈/止损：C(昨)→触发执行价(含卖出成本)

本版本新增逻辑：
- 利用 positions 中的 signal_date 列，标记每个交易日所属的“信号周期”；
- 维护 stopped_in_cycle 集合：记录在当前信号周期内已经触发过 TP/SL 的股票；
- 在该信号周期剩余日期中，这些股票即便在 positions 中仍有权重，也不会再次触发 BUY；
- 当 signal_date 变化（进入新的调仓周期）时，自动清空 stopped_in_cycle。
"""

from pathlib import Path
from typing import Optional, Dict, Set

import numpy as np
import pandas as pd

from backtest_rolling_config import BT_ROLL_CFG
from utils import load_calendar


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
    close_pivot = price_df["close"].unstack(0).sort_index()
    open_pivot  = price_df["open"].unstack(0).sort_index()
    high_pivot  = price_df["high"].unstack(0).sort_index()
    low_pivot   = price_df["low"].unstack(0).sort_index()
    return open_pivot, close_pivot, high_pivot, low_pivot


def _pivot_weights_for_day(positions: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    g = positions[positions["date"] == date]
    s = pd.Series(g["weight"].values.astype(float), index=g["stock"].astype(str).values)
    s = s[s != 0.0]
    return s


def main():
    cfg = BT_ROLL_CFG
    out_dir = cfg.backtest_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    weight_mode = str(getattr(cfg, "weight_mode", "score")).lower()
    pos_path = out_dir / f"positions_{cfg.run_name_out}_{weight_mode}.csv"
    if not pos_path.exists():
        raise FileNotFoundError(f"未找到持仓文件：{pos_path}")

    # 止盈/止损参数
    enable_stops = bool(getattr(cfg, "enable_intraweek_stops", False))
    tp_ratio = float(getattr(cfg, "tp_price_ratio", 0.06))
    sl_ratio = float(getattr(cfg, "sl_price_ratio", 0.06))

    # 读取日度持仓（包含 signal_date）
    positions = pd.read_csv(pos_path)
    positions["date"] = pd.to_datetime(positions["date"])
    if "signal_date" in positions.columns:
        positions["signal_date"] = pd.to_datetime(positions["signal_date"])
    else:
        # 兜底：若没有 signal_date 列，就直接用当日 date 作为 signal_date（相当于每日信号）
        positions["signal_date"] = positions["date"]

    positions = positions.sort_values(["date", "stock"])
    positions = positions[(positions["date"] >= pd.Timestamp(cfg.bt_start_date)) &
                          (positions["date"] <= pd.Timestamp(cfg.bt_end_date))]

    # 为每个交易日构造一个唯一的 signal_date（同一天所有股票应一致）
    # 若某天存在多个 signal_date（理论上不应该），取最早的一个
    day_signal_map = (
        positions.groupby("date")["signal_date"]
        .min()
        .to_dict()
    )

    # 读取日线
    price_df = pd.read_parquet(cfg.price_day_file)
    if not isinstance(price_df.index, pd.MultiIndex):
        price_df = price_df.set_index(["order_book_id", "date"]).sort_index()
    needed_cols = {"open", "close", "high", "low"}
    if not needed_cols.issubset(price_df.columns):
        raise RuntimeError("price_day_file 需包含 open/close/high/low 列")

    open_pv, close_pv, high_pv, low_pv = _build_pivots(price_df)

    # 交易日序列（以持仓中出现的日期为主，并与价格可用日期取交集）
    days = pd.DatetimeIndex(sorted(positions["date"].unique()))
    days = days.intersection(open_pv.index).intersection(close_pv.index)
    if len(days) == 0:
        print("[BTR-BT] 无有效交易日")
        return

    buy_cost, sell_cost = _cost_terms_from_cfg()

    recs = []
    stock_recs = []
    trade_recs = []  # 交易事件记录

    # 每只股票的“本轮入场价（裸价O）”与“是否已触发”
    entry_state: Dict[str, Dict[str, float]] = {}

    prev_w: Optional[pd.Series] = None  # 昨日权重

    # 新增：当前信号周期的 signal_date，以及本周期内已经被 TP/SL 的股票集合
    current_cycle_signal_date: Optional[pd.Timestamp] = None
    stopped_in_cycle: Set[str] = set()

    for i, d in enumerate(days):
        # ===== 0) 信号周期切换检测 =====
        sig_date_today = day_signal_map.get(d)

        if (current_cycle_signal_date is None) or (sig_date_today != current_cycle_signal_date):
            # 进入新的信号周期：清空“本周期已 stop 股票集合”
            current_cycle_signal_date = sig_date_today
            stopped_in_cycle.clear()

        # ===== 1) 目标权重与集合划分 =====
        w_today_target = _pivot_weights_for_day(positions, d)
        w_yest = prev_w

        set_today = set(w_today_target.index)
        set_yest = set() if (w_yest is None or w_yest.empty) else set(w_yest.index)

        C_t = close_pv.loc[d]
        O_t = open_pv.loc[d]
        H_t = high_pv.loc[d]
        L_t = low_pv.loc[d]
        d_prev = days[i - 1] if i > 0 else None
        C_prev = close_pv.loc[d_prev] if d_prev is not None else None

        # 原始新增集合
        A_add_raw = list(set_today - set_yest)
        # 过滤掉当前信号周期内已经被 TP/SL 的股票：这些股票在下一次信号前都不再允许新开仓
        A_add = [s for s in A_add_raw if s not in stopped_in_cycle]

        Y_all = list(set_yest)

        # ===== 2) 当日新入场：记录 entry_price（裸价O），并记录 BUY 交易 =====
        if A_add:
            for s in A_add:
                o_raw = float(O_t.get(s, np.nan))
                if np.isfinite(o_raw):
                    entry_state[s] = {"entry_price": o_raw, "triggered": 0.0}
                    exec_price = o_raw * (1.0 + buy_cost)
                    trade_recs.append({
                        "date": d, "stock": s, "action": "BUY",
                        "entry_price": o_raw, "exec_price": exec_price
                    })

        # ===== 3) 存量持有先检查止损->止盈 =====
        contrib_stops_total = 0.0
        if enable_stops and Y_all and C_prev is not None:
            for s in Y_all:
                state = entry_state.get(s)
                if state is None:
                    # 兜底：若状态丢失，用当日开盘作为入场基准
                    entry_state[s] = {"entry_price": float(O_t.get(s, np.nan)), "triggered": 0.0}
                    state = entry_state[s]
                if state["triggered"]:
                    continue
                # 仅对昨持有且今仍持有者检查
                if (s in set_yest) and (s in set_today) and np.isfinite(state["entry_price"]):
                    entry_price = float(state["entry_price"])
                    hi = float(H_t.get(s, np.nan))
                    lo = float(L_t.get(s, np.nan))
                    if not np.isfinite(hi) or not np.isfinite(lo):
                        continue

                    sl_price = entry_price * (1.0 - sl_ratio)
                    tp_price = entry_price * (1.0 + tp_ratio)

                    trig = None
                    exec_price = None
                    if lo <= sl_price:
                        trig = "SL"
                        exec_price = sl_price * (1.0 - sell_cost)
                    elif hi >= tp_price:
                        trig = "TP"
                        exec_price = tp_price * (1.0 - sell_cost)

                    if trig is not None:
                        # 用昨日权重全清仓，收益以 C_prev -> exec_price
                        w_prev = float(w_yest.get(s, 0.0)) if (w_yest is not None) else 0.0
                        if w_prev > 0 and C_prev is not None and np.isfinite(exec_price):
                            r = float(exec_price / float(C_prev.get(s, np.nan)) - 1.0)
                            contrib = w_prev * r
                            contrib_stops_total += contrib
                            stock_recs.append({
                                "date": d, "stock": s,
                                "weight": w_prev, "ret": r, "contribution": contrib
                            })
                            # 记录交易
                            trade_recs.append({
                                "date": d, "stock": s, "action": trig,
                                "entry_price": entry_price, "exec_price": exec_price
                            })
                        entry_state[s]["triggered"] = 1.0
                        # 关键：记录到本信号周期的 stop 集合中，后续该周期不再允许重新 BUY
                        stopped_in_cycle.add(s)

        # ===== 4) 其余路径：新增、存续、减持 =====
        # 对新增部分，再过滤掉“当日刚刚触发 stop 的股票”
        A_add_effective = [s for s in A_add if s not in stopped_in_cycle]
        S_keep = list((set_yest & set_today) - stopped_in_cycle)
        R_reduce = list((set_yest - set_today) - stopped_in_cycle)

        # 收益段
        ret_add = pd.Series(dtype=float)
        if A_add_effective:
            O_eff = (O_t.loc[A_add_effective] * (1.0 + buy_cost)).astype(float)
            ret_add = (C_t.loc[A_add_effective].astype(float) / O_eff - 1.0)

        ret_keep = pd.Series(dtype=float)
        if S_keep and C_prev is not None:
            ret_keep = (C_t.loc[S_keep].astype(float) / C_prev.loc[S_keep].astype(float) - 1.0)

        ret_reduce = pd.Series(dtype=float)
        if R_reduce and C_prev is not None:
            O_out = (O_t.loc[R_reduce] * (1.0 - sell_cost)).astype(float)
            ret_reduce = (O_out / C_prev.loc[R_reduce].astype(float) - 1.0)

        # 贡献
        contrib_add = (w_today_target.reindex(A_add_effective).fillna(0.0) * ret_add).sum() if len(ret_add) else 0.0
        contrib_keep = (w_today_target.reindex(S_keep).fillna(0.0) * ret_keep).sum() if len(ret_keep) else 0.0
        contrib_reduce = 0.0
        if len(ret_reduce):
            contrib_reduce = (w_yest.reindex(R_reduce).fillna(0.0) * ret_reduce).sum() if (w_yest is not None) else 0.0

        day_ret = float(contrib_stops_total + contrib_add + contrib_keep + contrib_reduce)

        # n_long：以日末目标剔除被 stop 的近似
        n_long = int(len(set_today - stopped_in_cycle))

        recs.append({
            "date": d,
            "ret_long": day_ret,
            "ret_short": 0.0,
            "ret_total": day_ret,
            "n_long": n_long
        })

        # ===== 5) 明细记录 =====
        if len(ret_add):
            for s, r in ret_add.items():
                w = float(w_today_target.get(s, 0.0))
                stock_recs.append({
                    "date": d, "stock": s,
                    "weight": w, "ret": float(r), "contribution": float(w * r)
                })
        if len(ret_keep):
            for s, r in ret_keep.items():
                w = float(w_today_target.get(s, 0.0))
                stock_recs.append({
                    "date": d, "stock": s,
                    "weight": w, "ret": float(r), "contribution": float(w * r)
                })
        if len(ret_reduce) and (w_yest is not None):
            for s, r in ret_reduce.items():
                w = float(w_yest.get(s, 0.0))
                stock_recs.append({
                    "date": d, "stock": s,
                    "weight": w, "ret": float(r), "contribution": float(w * r)
                })
                # 记录 SELL 交易（常规清仓或减仓的“卖出事件”）
                o_out_raw = float(O_t.get(s, np.nan))
                o_out = o_out_raw * (1.0 - sell_cost) if np.isfinite(o_out_raw) else np.nan
                entry_price = entry_state.get(s, {}).get("entry_price", np.nan)
                trade_recs.append({
                    "date": d, "stock": s, "action": "SELL",
                    "entry_price": entry_price, "exec_price": o_out
                })

        # ===== 6) 状态生命周期 =====
        to_del = []
        for s in set(set_yest | set_today):
            tgt_zero = (s not in set_today) or (s in stopped_in_cycle)
            if tgt_zero:
                if s in entry_state:
                    to_del.append(s)
        for s in to_del:
            entry_state.pop(s, None)

        prev_w = w_today_target

    # 输出净值
    nav_df = pd.DataFrame(recs).set_index("date").sort_index()
    nav_df["nav"] = (1.0 + nav_df["ret_total"]).cumprod()

    out_nav_csv = out_dir / f"nav_{cfg.run_name_out}.csv"
    ensure_dir(out_nav_csv)
    cols = ["ret_long", "ret_short", "ret_total", "nav", "n_long"]
    nav_df[cols].to_csv(out_nav_csv, float_format="%.8f")
    print(f"[BTR-BT] 已保存净值序列：{out_nav_csv}")

    # 输出股票收益明细
    stock_df = pd.DataFrame(stock_recs)
    if not stock_df.empty:
        stock_df["date"] = pd.to_datetime(stock_df["date"])
        stock_df = stock_df.sort_values(by=["date", "stock"]).reset_index(drop=True)
        out_stock_csv = out_dir / f"stock_returns_{cfg.run_name_out}.csv"
        stock_df.to_csv(out_stock_csv, index=False, float_format="%.8f")
        print(f"[BTR-BT] 已保存股票收益明细：{out_stock_csv}")

    # 输出交易记录
    trade_df = pd.DataFrame(trade_recs, columns=["date", "stock", "action", "entry_price", "exec_price"])
    if not trade_df.empty:
        trade_df["date"] = pd.to_datetime(trade_df["date"])
        trade_df = trade_df.sort_values(by=["date", "stock"]).reset_index(drop=True)
    out_trades_csv = out_dir / f"stock_trades_{cfg.run_name_out}.csv"
    ensure_dir(out_trades_csv)
    trade_df.to_csv(out_trades_csv, index=False, float_format="%.8f")
    print(f"[BTR-BT] 已保存交易记录：{out_trades_csv}")


if __name__ == "__main__":
    main()