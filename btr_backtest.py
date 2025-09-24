# coding: utf-8
"""
btr_backtest.py
读取 positions_{run_name}_{weight_mode}.csv + 原始日线，按 O2C 口径构建周度组合净值：
- 写 backtest_rolling/{run_name}/nav_{run_name}.csv（不带 weight_mode 后缀）
- 说明：持仓由 optimize_position.py 统一生成，本脚本不再做“先选后权重”。
- 成本与滑点：单股层计入（买万三/卖万八 + 对称滑点），组合层不重复扣。
- 可选周内止盈/止损：止损优先、可多次触发。
"""

import math
from pathlib import Path
from typing import Dict, Tuple

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


def compute_weekly_returns_o2c_basic(close_pivot: pd.DataFrame,
                                     fridays: pd.DatetimeIndex,
                                     open_pivot: pd.DataFrame) -> Tuple[Dict[pd.Timestamp, pd.Series], Dict[pd.Timestamp, pd.Timestamp]]:
    buy_cost, sell_cost = _cost_terms_from_cfg()
    avail = close_pivot.index
    out = {}
    start_date_map = {}

    for i in range(len(fridays) - 1):
        f0 = fridays[i]
        f1 = fridays[i + 1]
        i0 = avail[avail <= f0]
        i1 = avail[avail <= f1]
        if len(i0) == 0 or len(i1) == 0:
            continue
        d0 = i0[-1]
        d1 = i1[-1]
        pos = avail.get_indexer([d0])[0]
        if pos < 0 or pos + 1 >= len(avail):
            continue
        d_start = avail[pos + 1]
        if d_start > d1:
            continue
        try:
            O0 = open_pivot.loc[d_start]
            C_end = close_pivot.loc[d1]
        except KeyError:
            continue
        O0_eff = O0 * (1.0 + buy_cost)
        C_end_eff = C_end * (1.0 - sell_cost)
        ret = (C_end_eff / O0_eff - 1.0).replace([np.inf, -np.inf], np.nan)
        out[f0] = ret
        start_date_map[f0] = d_start
    return out, start_date_map


def compute_weekly_returns_with_stops(open_pivot: pd.DataFrame,
                                      high_pivot: pd.DataFrame,
                                      low_pivot: pd.DataFrame,
                                      close_pivot: pd.DataFrame,
                                      fridays: pd.DatetimeIndex,
                                      tp_price_ratio: float,
                                      sl_price_ratio: float,
                                      tp_sell_ratio: float,
                                      sl_sell_ratio: float,
                                      stop_priority: str,
                                      allow_multiple_triggers: bool) -> Dict[pd.Timestamp, pd.Series]:
    buy_cost, sell_cost = _cost_terms_from_cfg()
    avail = close_pivot.index
    weekly_ret: Dict[pd.Timestamp, pd.Series] = {}

    for i in range(len(fridays) - 1):
        f0 = fridays[i]
        f1 = fridays[i + 1]
        i0 = avail[avail <= f0]
        i1 = avail[avail <= f1]
        if len(i0) == 0 or len(i1) == 0:
            continue
        d0 = i0[-1]
        d1 = i1[-1]
        pos = avail.get_indexer([d0])[0]
        if pos < 0 or pos + 1 >= len(avail):
            continue
        d_start = avail[pos + 1]
        if d_start > d1:
            continue

        days_window = avail[(avail >= d_start) & (avail <= d1)]
        try:
            O0 = open_pivot.loc[d_start]
            C_end = close_pivot.loc[d1]
        except KeyError:
            continue

        stocks = open_pivot.columns
        O0_eff = O0 * (1.0 + buy_cost)
        P_tp_target = O0 * (1.0 + float(tp_price_ratio))
        P_sl_target = O0 * (1.0 - float(sl_price_ratio))

        seg_ret_sum = pd.Series(0.0, index=stocks, dtype=float)
        remain = pd.Series(1.0, index=stocks, dtype=float)

        for d in days_window:
            try:
                H = high_pivot.loc[d]
                L = low_pivot.loc[d]
            except KeyError:
                continue

            alive = (remain > 1e-12)

            # 止损优先
            hit_sl = alive & (L <= P_sl_target)
            if hit_sl.any():
                sell_ratio_sl = np.clip(sl_sell_ratio, 0.0, 1.0)
                delta_sl = sell_ratio_sl * remain
                delta_sl = delta_sl.where(hit_sl, other=0.0)
                P_sl_eff = P_sl_target * (1.0 - sell_cost)
                seg_ret_sum += delta_sl * (P_sl_eff / O0_eff - 1.0)
                remain = (remain - delta_sl).clip(lower=0.0)

            # 同日止盈仅对未止损者
            alive = (remain > 1e-12)
            hit_sl_today = (L <= P_sl_target)
            consider_tp = alive & (~hit_sl_today) & (H >= P_tp_target)
            if consider_tp.any():
                sell_ratio_tp = np.clip(tp_sell_ratio, 0.0, 1.0)
                delta_tp = sell_ratio_tp * remain
                delta_tp = delta_tp.where(consider_tp, other=0.0)
                P_tp_eff = P_tp_target * (1.0 - sell_cost)
                seg_ret_sum += delta_tp * (P_tp_eff / O0_eff - 1.0)
                remain = (remain - delta_tp).clip(lower=0.0)

            if not allow_multiple_triggers:
                pass

            if (remain <= 1e-12).all():
                break

        # 周末清算剩余
        C_end_eff = C_end * (1.0 - sell_cost)
        seg_ret_sum += remain * (C_end_eff / O0_eff - 1.0)

        weekly_ret[f0] = seg_ret_sum.replace([np.inf, -np.inf], np.nan)

    return weekly_ret


def build_nav_from_positions(positions: pd.DataFrame,
                             weekly_ret: Dict[pd.Timestamp, pd.Series],
                             long_weight: float) -> pd.DataFrame:
    """
    输入：
      - positions: 列至少包含 [date, stock, weight]，可有 score（忽略）
      - weekly_ret: dict[周五(date) -> Series(stock->下一周O2C收益，已含单股成本/滑点/止盈止损口径)]
    输出：
      - nav_df: index=date, cols=[ret_long, ret_short, ret_total, nav, n_long]
    """
    if positions.empty:
        return pd.DataFrame(columns=["ret_long", "ret_short", "ret_total", "nav", "n_long"])

    positions = positions.copy()
    positions["date"] = pd.to_datetime(positions["date"])
    dates = sorted(set(positions["date"]).intersection(weekly_ret.keys()))

    recs = []
    for d in dates:
        pos_d = positions[positions["date"] == d].copy()
        pos_d = pos_d.dropna(subset=["stock", "weight"])
        if pos_d.empty:
            recs.append({"date": d, "ret_long": 0.0, "ret_short": 0.0, "ret_total": 0.0, "n_long": 0})
            continue
        ret_s = weekly_ret[d]
        # 对齐到持仓股票
        pos_d = pos_d.merge(ret_s.rename("ret"), left_on="stock", right_index=True, how="inner")
        pos_d = pos_d.dropna(subset=["ret", "weight"])
        if pos_d.empty:
            recs.append({"date": d, "ret_long": 0.0, "ret_short": 0.0, "ret_total": 0.0, "n_long": 0})
            continue
        long_ret = float((pos_d["ret"].astype(float) * pos_d["weight"].astype(float)).sum())
        total = float(long_weight) * long_ret
        recs.append({"date": d, "ret_long": long_ret, "ret_short": 0.0, "ret_total": total, "n_long": int(pos_d.shape[0])})

    if not recs:
        return pd.DataFrame(columns=["ret_long", "ret_short", "ret_total", "nav", "n_long"])

    df_ret = pd.DataFrame(recs).set_index("date").sort_index()
    df_ret["nav"] = (1.0 + df_ret["ret_total"]).cumprod()
    return df_ret


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

    # 价格日线
    price_df = pd.read_parquet(cfg.price_day_file)
    if not isinstance(price_df.index, pd.MultiIndex):
        price_df = price_df.set_index(["order_book_id", "date"]).sort_index()
    need_cols = ["open", "high", "low", "close"]
    lack = [c for c in need_cols if c not in price_df.columns]
    if lack:
        raise RuntimeError(f"price_day_file 缺少必要列：{lack}")

    close_pivot = price_df["close"].unstack(0).sort_index()
    open_pivot  = price_df["open"].unstack(0).sort_index()
    high_pivot  = price_df["high"].unstack(0).sort_index()
    low_pivot   = price_df["low"].unstack(0).sort_index()

    cal = load_calendar(cfg.trading_day_file)
    fridays_all = weekly_fridays(cal)

    # 计算周度个股 O2C 收益（含成本/滑点；可选止盈止损）
    if getattr(cfg, "enable_intraweek_stops", False):
        weekly_ret_all = compute_weekly_returns_with_stops(
            open_pivot=open_pivot,
            high_pivot=high_pivot,
            low_pivot=low_pivot,
            close_pivot=close_pivot,
            fridays=fridays_all,
            tp_price_ratio=float(cfg.tp_price_ratio),
            sl_price_ratio=float(cfg.sl_price_ratio),
            tp_sell_ratio=float(cfg.tp_sell_ratio),
            sl_sell_ratio=float(cfg.sl_sell_ratio),
            stop_priority=str(getattr(cfg, "stop_priority", "SL_first")),
            allow_multiple_triggers=bool(getattr(cfg, "allow_multiple_triggers_per_week", True)),
        )
    else:
        weekly_ret_all, _ = compute_weekly_returns_o2c_basic(
            close_pivot=close_pivot,
            fridays=fridays_all,
            open_pivot=open_pivot
        )

    # 限定回测区间
    weekly_ret = {d: s for d, s in weekly_ret_all.items()
                  if (d >= pd.Timestamp(cfg.bt_start_date)) and (d <= pd.Timestamp(cfg.bt_end_date))}

    # 从 positions 聚合生成 NAV
    nav_df = build_nav_from_positions(positions, weekly_ret, long_weight=float(cfg.long_weight))
    if nav_df.empty:
        print("[BTR-BT] 未能生成回测净值（可能所有周都空仓或无可对齐的收益）。")
        return

    out_nav_csv = out_dir / f"nav_{cfg.run_name_out}.csv"
    ensure_dir(out_nav_csv)
    cols = ["ret_long", "ret_short", "ret_total", "nav", "n_long"]
    nav_df[cols].to_csv(out_nav_csv, float_format="%.8f")
    print(f"[BTR-BT] 已保存净值序列：{out_nav_csv}")


if __name__ == "__main__":
    main()