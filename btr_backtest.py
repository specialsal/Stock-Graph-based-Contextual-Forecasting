# coding: utf-8
"""
btr_backtest.py（日度版，支持一次性止盈/止损 + 冷静期 + 交易记录输出）

读取日度持仓 positions_{run_name_out}_{weight_mode}.csv + 原始日线，生成“日度净值”与成交记录：
- 新增：stock_trades_{run_name_out}.csv（列：date,stock,action,entry_price,exec_price）

成交事件定义：
- BUY: 当日目标从无到有，按开盘买入，exec_price=O*(1+买入成本)，entry_price=当日开盘裸价O
- SELL: 当日目标从有到无，且本轮未通过 TP/SL 平仓，按开盘卖出，exec_price=O*(1-卖出成本)，
         entry_price=该轮首次入场的开盘裸价
- TP: 止盈触发清仓，exec_price=entry_price*(1+tp_ratio)*(1-卖出成本)
- SL: 止损触发清仓，exec_price=entry_price*(1-sl_ratio)*(1-卖出成本)

当日收益拆分逻辑：
- 新增：O(今, 含买入成本)→C(今)
- 存续：C(昨)→C(今)
- 减仓/清仓：C(昨)→O(今, 含卖出成本)
- 止盈/止损：C(昨)→触发执行价(含卖出成本)；触发后当日不再参与其它路径

本版本额外规则（方案 A + 冷静期 5 天）：

1. 冷静期（5 个自然日）**强于持仓信号**：
   - 只要某只股票在某日触发 TP/SL，记 last_stop_date[s] = 该日；
   - 在接下来 5 天内（0 <= (today - last_stop_date).days <= 5）：
       - 即使 positions 给出正权重，也强制视为目标权重=0，不允许 BUY，不允许继续持仓；
       - 不会再触发 TP/SL。

2. 同一“持仓轮次”内，TP/SL 平仓后不再出现 SELL：
   - stop_closed[s] 标记本轮是否已通过 TP/SL 平仓；
   - 若 True，则后续常规调仓不再为这一轮生成 SELL 记录；
   - 下一次 BUY 该股票时，stop_closed[s] 重置为 False，开始新一轮。

3. 止盈/止损当天即完全平仓：
   - 当日对该股票只会出现一次 C_prev → exec_price(stop) 的收益记录；
   - 当日其他 O→C 或 C_prev→C 路径不再包含该股票；
   - 当日结束时从持仓状态中删除该股票。

4. 同一信号周期内不重复买回已止盈/止损的股票：
   - 利用 positions 里的 signal_date 列；
   - 每个 signal_date 为一个“信号周期”，在该周期内若某只股票被 TP/SL，则添加到 stopped_in_cycle；
   - 本周期内 stopped_in_cycle 中的股票即使 positions 给正权重也不会再 BUY；
   - 下个信号周期开始（signal_date 变化）时清空 stopped_in_cycle；
   - 注意：冷静期逻辑独立存在，跨周期仍然生效。
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
    """从配置中读入滑点 & 费率，返回买入成本、卖出成本（相对于裸价的比例）"""
    slp = float(BT_ROLL_CFG.slippage_bps) * 1e-4
    buy_fee = float(BT_ROLL_CFG.buy_fee_bps) * 1e-4
    sell_fee = float(BT_ROLL_CFG.sell_fee_bps) * 1e-4
    buy_cost = slp + buy_fee
    sell_cost = slp + sell_fee
    return buy_cost, sell_cost


def _build_pivots(price_df: pd.DataFrame):
    """把 (order_book_id, date) MultiIndex 的日线数据，透视成按日期索引的四个价格矩阵"""
    close_pivot = price_df["close"].unstack(0).sort_index()
    open_pivot  = price_df["open"].unstack(0).sort_index()
    high_pivot  = price_df["high"].unstack(0).sort_index()
    low_pivot   = price_df["low"].unstack(0).sort_index()
    return open_pivot, close_pivot, high_pivot, low_pivot


def _pivot_weights_for_day(positions: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    """从 positions 中取出某日的目标权重，返回 index=stock, value=weight 的 Series"""
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

    # 冷静期天数（自然日）
    cooldown_days = float(getattr(cfg, "cooldown_days", 1))

    # 读取日度持仓（要求包含 signal_date 列；若没有则用当日 date 代替）
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

    # 交易日序列：以持仓中出现的日期为主，并与价格日期取交集
    days = pd.DatetimeIndex(sorted(positions["date"].unique()))
    days = days.intersection(open_pv.index).intersection(close_pv.index)
    if len(days) == 0:
        print("[BTR-BT] 无有效交易日")
        return

    buy_cost, sell_cost = _cost_terms_from_cfg()

    recs = []         # 每日组合收益
    stock_recs = []   # 每日单票收益
    trade_recs = []   # 交易事件记录

    # 每只股票的“本轮入场价（裸价O）”与“是否在本轮已触发 TP/SL”
    entry_state: Dict[str, Dict[str, float]] = {}

    prev_w: Optional[pd.Series] = None  # 昨日目标权重（为计算减仓收益使用）

    # 当前信号周期的 signal_date，以及本周期内已经被 TP/SL 的股票集合
    current_cycle_signal_date: Optional[pd.Timestamp] = None
    stopped_in_cycle: Set[str] = set()

    # 方案 A：记录“本轮是否已通过 stop 平仓”，用于避免本轮再记 SELL
    stop_closed: Dict[str, bool] = {}

    # 冷静期：记录最近一次 TP/SL 日期（自然日）
    last_stop_date: Dict[str, pd.Timestamp] = {}

    # 冷静期辅助函数
    def _in_cooldown(stock: str, today: pd.Timestamp) -> bool:
        if stock not in last_stop_date:
            return False
        delta = (today - last_stop_date[stock]).days
        # 0 表示触发当天，随后 1~cooldown_days 为冷静期
        return 0 <= delta <= cooldown_days

    for i, d in enumerate(days):
        # ===== 0) 信号周期切换检测 =====
        sig_date_today = day_signal_map.get(d)

        if (current_cycle_signal_date is None) or (sig_date_today != current_cycle_signal_date):
            # 进入新的信号周期：清空“本周期已 stop 的股票集合”
            current_cycle_signal_date = sig_date_today
            stopped_in_cycle.clear()

        # ===== 1) 当日目标权重与集合划分（先不考虑冷静期） =====
        w_today_target = _pivot_weights_for_day(positions, d)
        # 更新昨日权重前，先把 0 权重的彻底移除 index（不参与 set_yest）
        if prev_w is not None:
            prev_w = prev_w[prev_w != 0.0]
        w_yest = prev_w
        # 昨日/今日股票集合（初始版本）
        set_today = set(w_today_target.index)
        set_yest = set() if (w_yest is None or w_yest.empty) else set(w_yest.index)

        C_t = close_pv.loc[d]
        O_t = open_pv.loc[d]
        H_t = high_pv.loc[d]
        L_t = low_pv.loc[d]

        d_prev = days[i - 1] if i > 0 else None
        C_prev = close_pv.loc[d_prev] if d_prev is not None else None

        # 初始的“所有昨日持有”集合
        Y_all = list(set_yest)

        # ===== 1.5) 冷静期在权重维度上强制砍仓 =====
        # 冷静期强于信号：无论 positions 给多大权重，只要在冷静期内，就视为目标权重=0 且不在 set_today 中
        for s in list(set_today):
            if _in_cooldown(s, d):
                w_today_target.loc[s] = 0.0
                set_today.remove(s)

        # 冷静期作用后，重新计算“今日目标有、昨日没有”的新增集合
        A_add_raw = list(set_today - set_yest)

        # ===== 2) 新增股票集合：再按“本周期已经 stop”过滤 =====
        # 当前信号周期里，只要一只股票被 TP/SL 过，本周期剩余时间不再允许 BUY
        A_add_cycle_filtered = [s for s in A_add_raw if s not in stopped_in_cycle]

        # 在这里可以不再额外按 _in_cooldown 过滤，因为冷静期已经在 set_today 中砍掉了
        A_add = A_add_cycle_filtered

        # 将“今日仍持有”的集合视为 Y_all 与 set_today 的交集（止损逻辑会进一步剔除）
        Y_all = list(set_yest)

        # ===== 3) 当日新入场：记录 entry_price（裸价O），并记录 BUY 交易 =====
        if A_add:
            for s in A_add:
                o_raw = float(O_t.get(s, np.nan))
                if not np.isfinite(o_raw):
                    continue

                # 记录本轮入场价与状态
                entry_state[s] = {"entry_price": o_raw, "triggered": 0.0}
                exec_price = o_raw * (1.0 + buy_cost)

                trade_recs.append({
                    "date": d,
                    "stock": s,
                    "action": "BUY",
                    "entry_price": o_raw,
                    "exec_price": exec_price
                })

                # 新一轮持仓开始，重置 stop_closed 状态
                stop_closed[s] = False

        # ===== 4) 存量持有先检查止损->止盈（只在 enable_stops 条件下） =====
        contrib_stops_total = 0.0

        if enable_stops and Y_all and C_prev is not None:
            # 注意：这里对“昨日持有 & 今日目标仍持有”的股票检查止损/止盈
            for s in list(Y_all):  # 用 list(...)，方便在循环中安全地 remove
                # 若该股票今天已经在冷静期内，被从 set_today 砍掉，则不参与止损检查
                if s not in set_yest or s not in set_today:
                    continue

                # 若这只股票在冷静期内，本轮已经 TP/SL 过了，理论上不该再进来
                if _in_cooldown(s, d):
                    # 冷静期内不再检查 TP/SL，且本日目标也被砍掉了
                    continue

                # === 显式取当日开盘裸价（关键修正点） ===
                o_raw = float(O_t.get(s, np.nan))
                if not np.isfinite(o_raw):
                    continue

                # entry_state 若不存在，兜底用当日开盘作为入场价
                state = entry_state.get(s)
                if state is None:
                    entry_state[s] = {"entry_price": o_raw, "triggered": 0.0}
                    state = entry_state[s]

                if state["triggered"]:
                    # 这一轮已经触发过 TP/SL，不再重复触发
                    continue

                entry_price = float(state["entry_price"])
                hi = float(H_t.get(s, np.nan))
                lo = float(L_t.get(s, np.nan))
                if not np.isfinite(hi) or not np.isfinite(lo) or not np.isfinite(entry_price):
                    continue

                sl_price = entry_price * (1.0 - sl_ratio)
                tp_price = entry_price * (1.0 + tp_ratio)

                trig = None
                exec_price = None

                # 先检查止损，再检查止盈（优先级可以根据需要调整）
                # 1) 止损逻辑
                if lo <= sl_price:
                    trig = "SL"
                    # 如果开盘就已经低于止损位，认为只能按开盘价成交（更悲观）
                    if o_raw <= sl_price:
                        exec_raw = o_raw
                    else:
                        # 否则视为盘中触发，按理想止损价成交
                        exec_raw = sl_price
                    exec_price = exec_raw * (1.0 - sell_cost)

                # 2) 止盈逻辑（只有在没有先触发止损的情况下才检查）
                elif hi >= tp_price:
                    trig = "TP"
                    # 如果开盘就已经高于止盈位，认为一开盘就能平掉，按开盘价
                    if o_raw >= tp_price:
                        exec_raw = o_raw
                    else:
                        # 否则视为盘中触发，按理想止盈价成交
                        exec_raw = tp_price
                    exec_price = exec_raw * (1.0 - sell_cost)

                if trig is not None:
                    # === 4.1) 用昨日权重全清仓，收益以 C_prev -> exec_price ===
                    w_prev = float(w_yest.get(s, 0.0)) if (w_yest is not None) else 0.0
                    if w_prev > 0 and np.isfinite(exec_price):
                        base = float(C_prev.get(s, np.nan))
                        if np.isfinite(base) and base > 0:
                            r = float(exec_price / base - 1.0)
                            contrib = w_prev * r
                            contrib_stops_total += contrib

                            stock_recs.append({
                                "date": d,
                                "stock": s,
                                "weight": w_prev,
                                "ret": r,
                                "contribution": contrib
                            })

                            # 记录 TP/SL 交易
                            trade_recs.append({
                                "date": d,
                                "stock": s,
                                "action": trig,
                                "entry_price": entry_price,
                                "exec_price": exec_price
                            })

                    # === 4.2) 状态标记 ===
                    entry_state[s]["triggered"] = 1.0
                    stopped_in_cycle.add(s)
                    stop_closed[s] = True
                    last_stop_date[s] = d

                    # === 4.3) 关键：当天从“持仓集合”和目标权重中彻底移除这只股票 ===
                    if s in set_today:
                        set_today.remove(s)
                    if s in set_yest:
                        set_yest.remove(s)
                    if s in Y_all:
                        Y_all.remove(s)
                    if s in w_today_target.index:
                        w_today_target.loc[s] = 0.0

                    # 同时从 entry_state 中删掉，以结束本轮；冷静期 & stopped_in_cycle 会防止立刻买回
                    if s in entry_state:
                        entry_state.pop(s, None)

        # ===== 5) 其余路径：新增、存续、减持（止损股票已被剔除） =====
        # 重新计算集合：此时 set_today / set_yest 已经剔除了当日触发 TP/SL 的股票
        A_add_effective = list(set_today - set_yest)  # 当日仍被视为新增
        S_keep = list(set_yest & set_today)           # 昨日持有且今日仍持有
        R_reduce = list(set_yest - set_today)         # 昨日持有但今日不再持有

        # 新增收益：O(今, 含买入成本) -> C(今)
        ret_add = pd.Series(dtype=float)
        if A_add_effective:
            o_eff = (O_t.loc[A_add_effective] * (1.0 + buy_cost)).astype(float)
            ret_add = (C_t.loc[A_add_effective].astype(float) / o_eff - 1.0)

        # 存续收益：C(昨) -> C(今)
        ret_keep = pd.Series(dtype=float)
        if S_keep and C_prev is not None:
            ret_keep = (C_t.loc[S_keep].astype(float) / C_prev.loc[S_keep].astype(float) - 1.0)

        # 减持/清仓收益：C(昨) -> O(今, 含卖出成本)
        ret_reduce = pd.Series(dtype=float)
        if R_reduce and C_prev is not None:
            o_out = (O_t.loc[R_reduce] * (1.0 - sell_cost)).astype(float)
            ret_reduce = (o_out / C_prev.loc[R_reduce].astype(float) - 1.0)

        # ===== 6) 计算日度组合收益 =====
        contrib_add = 0.0
        if len(ret_add):
            contrib_add = (w_today_target.reindex(A_add_effective).fillna(0.0) * ret_add).sum()

        contrib_keep = 0.0
        if len(ret_keep):
            contrib_keep = (w_today_target.reindex(S_keep).fillna(0.0) * ret_keep).sum()

        contrib_reduce = 0.0
        if len(ret_reduce) and (w_yest is not None):
            contrib_reduce = (w_yest.reindex(R_reduce).fillna(0.0) * ret_reduce).sum()

        day_ret = float(contrib_stops_total + contrib_add + contrib_keep + contrib_reduce)

        # n_long：以日末目标剔除被 stop 的近似（set_today 已不含当日触发 stop 的股票）
        n_long = int(len(set_today))

        recs.append({
            "date": d,
            "ret_long": day_ret,
            "ret_short": 0.0,
            "ret_total": day_ret,
            "n_long": n_long
        })

        # ===== 7) 明细记录（新增 / 存续 / 减持） =====
        if len(ret_add):
            for s, r in ret_add.items():
                w = float(w_today_target.get(s, 0.0))
                stock_recs.append({
                    "date": d,
                    "stock": s,
                    "weight": w,
                    "ret": float(r),
                    "contribution": float(w * r)
                })
        if len(ret_keep):
            for s, r in ret_keep.items():
                w = float(w_today_target.get(s, 0.0))
                stock_recs.append({
                    "date": d,
                    "stock": s,
                    "weight": w,
                    "ret": float(r),
                    "contribution": float(w * r)
                })
        if len(ret_reduce) and (w_yest is not None):
            for s, r in ret_reduce.items():
                w = float(w_yest.get(s, 0.0))
                stock_recs.append({
                    "date": d,
                    "stock": s,
                    "weight": w,
                    "ret": float(r),
                    "contribution": float(w * r)
                })

                # 方案 A：若本轮曾经通过 stop 平仓，则不再记录 SELL（只记 TP/SL）
                if stop_closed.get(s, False):
                    continue

                # 记录常规 SELL 交易
                o_out_raw = float(O_t.get(s, np.nan))
                o_out = o_out_raw * (1.0 - sell_cost) if np.isfinite(o_out_raw) else np.nan
                entry_price = entry_state.get(s, {}).get("entry_price", np.nan)

                trade_recs.append({
                    "date": d,
                    "stock": s,
                    "action": "SELL",
                    "entry_price": entry_price,
                    "exec_price": o_out
                })

                # 本轮结束，可以删除状态（也可以留到下一轮重置）
                if s in entry_state:
                    entry_state.pop(s, None)
                stop_closed[s] = False  # 卖出后本轮自然结束

        # ===== 8) 状态生命周期清理 =====
        # 对于今日既不在 set_today 也不在 set_yest 的股票，可以清理 entry_state
        to_del = []
        alive_stocks = set_today | set_yest
        for s in list(entry_state.keys()):
            if s not in alive_stocks:
                to_del.append(s)
        for s in to_del:
            entry_state.pop(s, None)
            # last_stop_date 保留，用于未来冷静期判断
            # stop_closed 在无仓位时可以保留或删除，这里保留，下一次 BUY 会重置为 False

        # 更新昨日权重
        prev_w = w_today_target

    # ===== 9) 输出净值序列 =====
    nav_df = pd.DataFrame(recs).set_index("date").sort_index()
    nav_df["nav"] = (1.0 + nav_df["ret_total"]).cumprod()

    out_nav_csv = out_dir / f"nav_{cfg.run_name_out}.csv"
    ensure_dir(out_nav_csv)
    cols = ["ret_long", "ret_short", "ret_total", "nav", "n_long"]
    nav_df[cols].to_csv(out_nav_csv, float_format="%.8f")
    print(f"[BTR-BT] 已保存净值序列：{out_nav_csv}")

    # ===== 10) 输出股票收益明细 =====
    stock_df = pd.DataFrame(stock_recs)
    if not stock_df.empty:
        stock_df["date"] = pd.to_datetime(stock_df["date"])
        stock_df = stock_df.sort_values(by=["date", "stock"]).reset_index(drop=True)
        out_stock_csv = out_dir / f"stock_returns_{cfg.run_name_out}.csv"
        stock_df.to_csv(out_stock_csv, index=False, float_format="%.8f")
        print(f"[BTR-BT] 已保存股票收益明细：{out_stock_csv}")

    # ===== 11) 输出交易记录 =====
    trade_df = pd.DataFrame(trade_recs, columns=["date", "stock", "action", "entry_price", "exec_price"])
    if not trade_df.empty:
        trade_df["date"] = pd.to_datetime(trade_df["date"])
        trade_df = trade_df.sort_values(by=["date", "stock", "action"]).reset_index(drop=True)
    out_trades_csv = out_dir / f"stock_trades_{cfg.run_name_out}.csv"
    ensure_dir(out_trades_csv)
    trade_df.to_csv(out_trades_csv, index=False, float_format="%.8f")
    print(f"[BTR-BT] 已保存交易记录：{out_trades_csv}")


if __name__ == "__main__":
    main()