# coding: utf-8
"""
stop_rules.py

封装各种止盈/止损规则，支持多规则叠加执行。

已实现：
- rule_fixed_tp_sl: 固定止盈/止损
- rule_trailing_sl: 跟踪止损（价格先超过某个比例才开启跟踪）
- run_stop_rules: 规则引擎，按配置列表顺序执行多个规则
"""

from typing import Dict, Set, List, Tuple, Optional

import numpy as np
import pandas as pd

from backtest_rolling_config import BT_ROLL_CFG


# ========= 通用：冷静期判断 =========

def in_cooldown(
    stock: str,
    today: pd.Timestamp,
    last_stop_date: Dict[str, pd.Timestamp],
    cooldown_days: float,
) -> bool:
    """判断某只股票今天是否处于冷静期（含触发日）"""
    if stock not in last_stop_date:
        return False
    delta = (today - last_stop_date[stock]).days
    return 0 <= delta <= cooldown_days


# ========= 规则 1：固定止盈/止损 =========

def rule_fixed_tp_sl(
    d: pd.Timestamp,
    enable_stops: bool,
    tp_ratio: float,
    sl_ratio: float,
    cooldown_days: float,
    set_today: Set[str],
    set_yest: Set[str],
    Y_all: List[str],
    w_yest: Optional[pd.Series],
    O_t: pd.Series,
    H_t: pd.Series,
    L_t: pd.Series,
    C_prev: Optional[pd.Series],
    buy_cost: float,
    sell_cost: float,
    entry_state: Dict[str, Dict[str, float]],
    stopped_in_cycle: Set[str],
    stop_closed: Dict[str, bool],
    last_stop_date: Dict[str, pd.Timestamp],
    stock_recs: List[Dict],
    trade_recs: List[Dict],
) -> Tuple[float, Set[str], Set[str], List[str]]:
    """
    固定止盈/止损规则。
    不直接改 w_today_target，只通过集合把被 stop 的股票剔除。
    """
    contrib_stops_total = 0.0

    if (not enable_stops) or (not Y_all) or (C_prev is None):
        return contrib_stops_total, set_today, set_yest, Y_all

    for s in list(Y_all):
        # 必须：昨日持有 + 今日仍为目标持有
        if s not in set_yest or s not in set_today:
            continue

        # 冷静期内不检查 TP/SL
        if in_cooldown(s, d, last_stop_date, cooldown_days):
            continue

        # 当日开盘裸价
        o_raw = float(O_t.get(s, np.nan))
        if not np.isfinite(o_raw):
            continue

        # entry_state 若不存在，兜底用当日开盘作为入场价
        state = entry_state.get(s)
        if state is None:
            entry_state[s] = {"entry_price": o_raw, "triggered": 0.0}
            state = entry_state[s]

        if state.get("triggered", 0.0):
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

        # 先止损后止盈
        if lo <= sl_price:
            trig = "SL"
            if o_raw <= sl_price:
                exec_raw = o_raw
            else:
                exec_raw = sl_price
            exec_price = exec_raw * (1.0 - sell_cost)
        elif hi >= tp_price:
            trig = "TP"
            if o_raw >= tp_price:
                exec_raw = o_raw
            else:
                exec_raw = tp_price
            exec_price = exec_raw * (1.0 - sell_cost)

        if trig is None:
            continue

        # 昨日权重全平：C_prev -> exec_price
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
                    "contribution": contrib,
                })

                trade_recs.append({
                    "date": d,
                    "stock": s,
                    "action": trig,
                    "entry_price": entry_price,
                    "exec_price": exec_price,
                })

        # 状态与集合更新
        entry_state[s]["triggered"] = 1.0
        stopped_in_cycle.add(s)
        stop_closed[s] = True
        last_stop_date[s] = d

        if s in set_today:
            set_today.remove(s)
        if s in set_yest:
            set_yest.remove(s)
        if s in Y_all:
            Y_all.remove(s)

        if s in entry_state:
            entry_state.pop(s, None)

    return contrib_stops_total, set_today, set_yest, Y_all


# ========= 规则 2：跟踪止损 =========

def rule_trailing_sl(
    d: pd.Timestamp,
    enable_stops: bool,
    trailing_sl_start_ratio: float,
    trailing_sl_drawdown: float,
    cooldown_days: float,
    set_today: Set[str],
    set_yest: Set[str],
    Y_all: List[str],
    w_yest: Optional[pd.Series],
    O_t: pd.Series,
    H_t: pd.Series,
    L_t: pd.Series,
    C_prev: Optional[pd.Series],
    buy_cost: float,
    sell_cost: float,
    entry_state: Dict[str, Dict[str, float]],
    trailing_state: Dict[str, Dict[str, float]],
    stopped_in_cycle: Set[str],
    stop_closed: Dict[str, bool],
    last_stop_date: Dict[str, pd.Timestamp],
    stock_recs: List[Dict],
    trade_recs: List[Dict],
) -> Tuple[float, Set[str], Set[str], List[str]]:
    """
    跟踪止损规则（按时间先后拆分为“开盘检查”和“盘中检查”）：

    语义约定：
    - entry_price：入场价，来自 entry_state[s]["entry_price"]
    - tracking_started：一旦某天最高价相对入场价涨幅 >= trailing_sl_start_ratio，则置为 True
    - highest_price：始终保存「最近一个交易日结束时的最高价」，即写回当天状态后
      第二天开盘拿到的 highest_price 是「截至昨天收盘为止的最高价」，不含今天的 high。
    - 开盘检查：
        以昨日至今最高价 highest_prev 为基准，计算 trailing 线：
            trail_sl_open = highest_prev * (1 - trailing_sl_drawdown)
        若 o_raw <= trail_sl_open，则视作“开盘即触发”，按 o_raw（扣成本后）成交，
        当天后续最高/最低价格不再影响（因为已经卖掉了）。
    - 盘中检查：
        若开盘未触发，才将当日最高价并入
            highest_today = max(highest_prev, hi)
        以 highest_today 为基准计算 trailing 线：
            trail_sl_intra = highest_today * (1 - trailing_sl_drawdown)
        若 lo <= trail_sl_intra，视作“盘中曾触发”，
        成交价按 trail_sl_intra（扣成本后）计算。
    """
    contrib_stops_total = 0.0

    # 顶层拦截：没开止损 / 没有股票 / 没有昨收，直接退出
    if (not enable_stops) or (not Y_all) or (C_prev is None):
        return contrib_stops_total, set_today, set_yest, Y_all

    for s in list(Y_all):
        # 必须：昨持 + 今仍目标持有
        if s not in set_yest or s not in set_today:
            continue

        # 冷静期内不做
        if in_cooldown(s, d, last_stop_date, cooldown_days):
            continue

        o_raw = float(O_t.get(s, np.nan))
        hi = float(H_t.get(s, np.nan))
        lo = float(L_t.get(s, np.nan))
        if not np.isfinite(o_raw) or not np.isfinite(hi) or not np.isfinite(lo):
            continue

        # entry_state 若不存在，兜底用当日开盘作为入场价
        e_state = entry_state.get(s)
        if e_state is None:
            entry_state[s] = {"entry_price": o_raw, "triggered": 0.0}
            e_state = entry_state[s]

        # 本轮（规则链）已被其它规则 stop 过，则不再处理
        if e_state.get("triggered", 0.0):
            continue

        entry_price = float(e_state["entry_price"])
        if not np.isfinite(entry_price) or entry_price <= 0:
            continue

        # 初始化 / 读取 trailing_state
        t_state = trailing_state.get(s)
        if t_state is None:
            trailing_state[s] = {
                "highest_price": entry_price,  # 启动前，最高价从 entry_price 开始
                "tracking_started": 0.0,
            }
            t_state = trailing_state[s]

        # 注意：这里的 highest_prev 是「截至昨天收盘的最高价」
        highest_prev = float(t_state.get("highest_price", entry_price))
        tracking_started = bool(t_state.get("tracking_started", 0.0))

        # ========== 第一步：判断是否需要启动 tracking ==========
        # 使用「当日最高价相对 entry_price 的涨幅」作为启动条件
        if (not tracking_started):
            up_ratio = hi / entry_price - 1.0
            if up_ratio >= trailing_sl_start_ratio:
                tracking_started = True

        # 若尚未启动 tracking，则只更新状态并继续
        if not tracking_started:
            trailing_state[s]["highest_price"] = highest_prev
            trailing_state[s]["tracking_started"] = 0.0
            continue

        # 走到这里：tracking 已经开启
        # ========== 第二步：开盘检查（使用昨日最高价 highest_prev） ==========
        trail_sl_price_open = highest_prev * (1.0 - trailing_sl_drawdown)
        if o_raw <= trail_sl_price_open:
            # 开盘即触发：按开盘成交（更悲观）
            exec_raw = o_raw
            exec_price = exec_raw * (1.0 - sell_cost)

            # 昨日权重全平：C_prev -> exec_price
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
                        "contribution": contrib,
                    })

                    trade_recs.append({
                        "date": d,
                        "stock": s,
                        "action": "TRAIL_SL",
                        "entry_price": entry_price,
                        "exec_price": exec_price,
                    })

            # 状态与集合更新
            e_state["triggered"] = 1.0
            stopped_in_cycle.add(s)
            stop_closed[s] = True
            last_stop_date[s] = d

            if s in set_today:
                set_today.remove(s)
            if s in set_yest:
                set_yest.remove(s)
            if s in Y_all:
                Y_all.remove(s)

            if s in entry_state:
                entry_state.pop(s, None)
            if s in trailing_state:
                trailing_state.pop(s, None)

            # 开盘已经卖出，今天后续高低价不再参与
            continue

        # ========== 第三步：盘中检查 ==========
        # 开盘没有触发，才允许把当日最高价并入最高价
        highest_today = max(highest_prev, hi)

        # 以 highest_today 为基准，计算盘中回撤比例（lo 是否曾经触及阈值）
        drawdown_from_high = 1.0 - (lo / highest_today)
        if drawdown_from_high < trailing_sl_drawdown:
            # 回撤尚不足以触发止损，写回最新的最高价与 tracking 状态
            trailing_state[s]["highest_price"] = highest_today
            trailing_state[s]["tracking_started"] = 1.0
            continue

        # 回撤达到阈值：视作盘中曾跌破 trailing 线
        trail_sl_price_intra = highest_today * (1.0 - trailing_sl_drawdown)

        # 按策略语义：既然开盘未触发（前面 if 已排除 o_raw <= trail_sl_price_open），
        # 这里可以直接使用 trail_sl_price_intra 作为成交基准价
        exec_raw = trail_sl_price_intra
        exec_price = exec_raw * (1.0 - sell_cost)

        # 按昨日权重全平：C_prev -> exec_price
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
                    "contribution": contrib,
                })

                trade_recs.append({
                    "date": d,
                    "stock": s,
                    "action": "TRAIL_SL",
                    "entry_price": entry_price,
                    "exec_price": exec_price,
                })

        # 状态与集合更新（与开盘触发完全相同）
        e_state["triggered"] = 1.0
        stopped_in_cycle.add(s)
        stop_closed[s] = True
        last_stop_date[s] = d

        if s in set_today:
            set_today.remove(s)
        if s in set_yest:
            set_yest.remove(s)
        if s in Y_all:
            Y_all.remove(s)

        if s in entry_state:
            entry_state.pop(s, None)
        if s in trailing_state:
            trailing_state.pop(s, None)

    return contrib_stops_total, set_today, set_yest, Y_all


# ========= 规则引擎：多规则叠加 =========

def run_stop_rules(
    rules: List[str],
    d: pd.Timestamp,
    enable_stops: bool,
    tp_ratio: float,
    sl_ratio: float,
    trailing_sl_start_ratio: float,
    trailing_sl_drawdown: float,
    cooldown_days: float,
    set_today: Set[str],
    set_yest: Set[str],
    Y_all: List[str],
    w_yest: Optional[pd.Series],
    O_t: pd.Series,
    H_t: pd.Series,
    L_t: pd.Series,
    C_prev: Optional[pd.Series],
    buy_cost: float,
    sell_cost: float,
    entry_state: Dict[str, Dict[str, float]],
    trailing_state: Dict[str, Dict[str, float]],
    stopped_in_cycle: Set[str],
    stop_closed: Dict[str, bool],
    last_stop_date: Dict[str, pd.Timestamp],
    stock_recs: List[Dict],
    trade_recs: List[Dict],
) -> Tuple[float, Set[str], Set[str], List[str]]:
    """
    按顺序执行多个止盈/止损规则，返回：
    - 所有规则产生的总贡献 contrib_stops_total
    - 更新后的 set_today / set_yest / Y_all
    """
    if (not enable_stops) or (not rules) or (not Y_all) or (C_prev is None):
        return 0.0, set_today, set_yest, Y_all

    total_contrib = 0.0
    rules_lower = [str(r).lower() for r in rules]

    for r in rules_lower:
        if not Y_all:
            break

        if r == "fixed_tp_sl":
            contrib, set_today, set_yest, Y_all = rule_fixed_tp_sl(
                d=d,
                enable_stops=enable_stops,
                tp_ratio=tp_ratio,
                sl_ratio=sl_ratio,
                cooldown_days=cooldown_days,
                set_today=set_today,
                set_yest=set_yest,
                Y_all=Y_all,
                w_yest=w_yest,
                O_t=O_t,
                H_t=H_t,
                L_t=L_t,
                C_prev=C_prev,
                buy_cost=buy_cost,
                sell_cost=sell_cost,
                entry_state=entry_state,
                stopped_in_cycle=stopped_in_cycle,
                stop_closed=stop_closed,
                last_stop_date=last_stop_date,
                stock_recs=stock_recs,
                trade_recs=trade_recs,
            )
            total_contrib += contrib

        elif r == "trailing_sl":
            contrib, set_today, set_yest, Y_all = rule_trailing_sl(
                d=d,
                enable_stops=enable_stops,
                trailing_sl_start_ratio=trailing_sl_start_ratio,
                trailing_sl_drawdown=trailing_sl_drawdown,
                cooldown_days=cooldown_days,
                set_today=set_today,
                set_yest=set_yest,
                Y_all=Y_all,
                w_yest=w_yest,
                O_t=O_t,
                H_t=H_t,
                L_t=L_t,
                C_prev=C_prev,
                buy_cost=buy_cost,
                sell_cost=sell_cost,
                entry_state=entry_state,
                trailing_state=trailing_state,
                stopped_in_cycle=stopped_in_cycle,
                stop_closed=stop_closed,
                last_stop_date=last_stop_date,
                stock_recs=stock_recs,
                trade_recs=trade_recs,
            )
            total_contrib += contrib

        else:
            # 未知规则名：忽略
            continue

    return total_contrib, set_today, set_yest, Y_all