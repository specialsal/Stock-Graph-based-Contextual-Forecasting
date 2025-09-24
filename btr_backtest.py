# coding: utf-8
"""
btr_backtest.py
读取 backtest_rolling/{run_name}/predictions_filtered_{run_name}.parquet + 原始日线，
按 Top-K 先选后权重（等权 / 按分数加权），构建周度组合，输出：
- backtest_rolling/{run_name}/positions_{run_name}.csv
- backtest_rolling/{run_name}/nav_{run_name}.csv

本版要点：
- 收益口径为 O2C（周五信号 -> 下周开盘买 -> 下周五收盘清算）
- 可选“周内止盈/止损”，含多次触发、止损优先
- 成本与滑点：单股层面计入，非对称手续费（买万三/卖万八）+ 对称滑点（万五）
  买入有效价 O0_eff = O0 * (1 + slippage + buy_fee)
  卖出有效价 P_eff = P * (1 - slippage - sell_fee)
- 组合层不再重复扣成本
"""

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from backtest_rolling_config import BT_ROLL_CFG
from utils import load_calendar, weekly_fridays
from config import CFG


def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def _cost_terms_from_cfg():
    # 返回买卖两侧的单边综合成本（含滑点+手续费），单位转成小数
    slp = float(BT_ROLL_CFG.slippage_bps) * 1e-4
    buy_fee = float(BT_ROLL_CFG.buy_fee_bps) * 1e-4
    sell_fee = float(BT_ROLL_CFG.sell_fee_bps) * 1e-4
    buy_cost = slp + buy_fee
    sell_cost = slp + sell_fee
    return buy_cost, sell_cost


def compute_weekly_returns_o2c_basic(close_pivot: pd.DataFrame,
                                     fridays: pd.DatetimeIndex,
                                     open_pivot: pd.DataFrame) -> Tuple[Dict[pd.Timestamp, pd.Series], Dict[pd.Timestamp, pd.Timestamp]]:
    """
    O2C（不含止盈止损），在单股层计入非对称成本：
      O0_eff = O0 * (1 + buy_cost)
      C_end_eff = C_end * (1 - sell_cost)
      ret = C_end_eff / O0_eff - 1
    """
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
    """
    含周内止盈/止损的 O2C 扩展版，单股层计入非对称成本。
    """
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
        ret_series = pd.Series(index=stocks, dtype=float)
        ret_series[:] = np.nan

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

            # 同日止盈仅对未触发止损者
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


def build_portfolio_with_weights(scores: pd.DataFrame,
                                 weekly_ret: Dict[pd.Timestamp, pd.Series],
                                 mode: str,
                                 top_pct: float,
                                 max_n: int,
                                 min_n: int,
                                 long_w: float,
                                 weight_mode: str = "equal",
                                 filter_negative_scores_long: bool = True,
                                 include_last_no_return_week: bool = True):
    """
    返回：
      - nav_df: index=date, cols=[ret_long, ret_short, ret_total, nav, n_long]
      - pos_df: 每周持仓与权重明细（date, stock, weight, score, next_week_ret）
    说明：
      - 单股收益已计入成本，组合层不再额外扣。
    """
    if scores.empty:
        return pd.DataFrame(), pd.DataFrame()

    dates_with_ret = sorted(set(scores["date"]).intersection(weekly_ret.keys()))
    recs = []
    pos_rows = []

    for d in dates_with_ret:
        df_d = scores[scores["date"] == d].copy()
        ret_s = weekly_ret[d]
        df_d = df_d.merge(ret_s.rename("ret"), left_on="stock", right_index=True, how="inner")
        df_d = df_d.dropna(subset=["ret", "score"])
        if len(df_d) < min_n:
            recs.append({"date": d, "ret_long": 0.0, "ret_short": 0.0, "ret_total": 0.0, "n_long": 0})
            continue

        df_d = df_d.sort_values(["score", "stock"], ascending=[False, True])
        n = len(df_d)
        n_target = max(0, min(int(math.floor(n * top_pct)), max_n))
        if n_target == 0 and n >= min_n:
            n_target = min(min_n, n)
        df_sel = df_d.head(n_target).copy()

        if filter_negative_scores_long:
            df_sel = df_sel[df_sel["score"] > 0]

        if len(df_sel) < min_n:
            recs.append({"date": d, "ret_long": 0.0, "ret_short": 0.0, "ret_total": 0.0, "n_long": 0})
            continue

        if weight_mode == "equal":
            w = np.full(len(df_sel), 1.0 / len(df_sel), dtype=float)
        else:
            s = df_sel["score"].values.astype(float)
            s = np.clip(s, 0.0, None)
            ssum = s.sum()
            w = np.full(len(df_sel), 1.0 / len(df_sel), dtype=float) if ssum <= 0 else s / ssum

        df_sel = df_sel.assign(weight=w)

        long_ret = float((df_sel["ret"] * df_sel["weight"]).sum())
        total = long_w * long_ret

        recs.append({"date": d, "ret_long": long_ret, "ret_short": 0.0, "ret_total": total, "n_long": len(df_sel)})
        for _, r in df_sel.iterrows():
            pos_rows.append({
                "date": d,
                "stock": r["stock"],
                "weight": float(r["weight"]),
                "score": float(r["score"]),
                "next_week_ret": float(r["ret"])
            })

    if include_last_no_return_week:
        all_score_dates = sorted(set(scores["date"]))
        if all_score_dates:
            last_d = all_score_dates[-1]
            if last_d not in weekly_ret:
                df_last = scores[scores["date"] == last_d].copy()
                df_last = df_last.dropna(subset=["score"])
                if len(df_last) >= min_n:
                    df_last = df_last.sort_values(["score", "stock"], ascending=[False, True])
                    n = len(df_last)
                    n_target = max(0, min(int(math.floor(n * top_pct)), max_n))
                    if n_target == 0 and n >= min_n:
                        n_target = min(min_n, n)
                    df_sel = df_last.head(n_target).copy()
                    if filter_negative_scores_long:
                        df_sel = df_sel[df_sel["score"] > 0]
                    if len(df_sel) >= min_n:
                        if weight_mode == "equal":
                            w = np.full(len(df_sel), 1.0 / len(df_sel), dtype=float)
                        else:
                            s = df_sel["score"].values.astype(float)
                            s = np.clip(s, 0.0, None)
                            ssum = s.sum()
                            w = np.full(len(df_sel), 1.0 / len(df_sel), dtype=float) if ssum <= 0 else s / ssum
                        df_sel = df_sel.assign(weight=w)
                        for _, r in df_sel.iterrows():
                            pos_rows.append({
                                "date": last_d,
                                "stock": r["stock"],
                                "weight": float(r["weight"]),
                                "score": float(r["score"]),
                                "next_week_ret": float("nan")
                            })

    if not recs:
        return pd.DataFrame(columns=["ret_long", "ret_short", "ret_total", "nav", "n_long"]), pd.DataFrame(pos_rows)

    df_ret = pd.DataFrame(recs).set_index("date").sort_index()
    df_ret["nav"] = (1.0 + df_ret["ret_total"]).cumprod()

    pos_df = pd.DataFrame(pos_rows)
    if not pos_df.empty:
        pos_df = pos_df.sort_values(["date", "weight"], ascending=[True, False])

    return df_ret, pos_df


def main():
    cfg = BT_ROLL_CFG
    out_dir = cfg.backtest_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 读取打分
    pred_path = out_dir / f"predictions_filtered_{cfg.run_name_out}.parquet"
    if not pred_path.exists():
        alt = out_dir / f"predictions_filtered_{cfg.run_name}.parquet"
        if alt.exists():
            pred_path = alt
    if not pred_path.exists():
        raise FileNotFoundError(f"未找到池内打分文件：{pred_path}")

    scores = pd.read_parquet(pred_path)
    scores = scores[(scores["date"] >= pd.Timestamp(cfg.bt_start_date)) & (scores["date"] <= pd.Timestamp(cfg.bt_end_date))]
    if scores.empty:
        raise RuntimeError("回测区间内打分为空")

    # 日线透视表
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

    # 周度个股收益
    if getattr(cfg, "enable_intraweek_stops", True):
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

    weekly_ret = {d: s for d, s in weekly_ret_all.items()
                  if (d >= pd.Timestamp(cfg.bt_start_date)) and (d <= pd.Timestamp(cfg.bt_end_date))}

    # 组合（权重与过滤从配置读取）
    nav_df, pos_df = build_portfolio_with_weights(
        scores, weekly_ret,
        mode=cfg.mode,
        top_pct=cfg.top_pct,
        max_n=cfg.max_n_stocks,
        min_n=cfg.min_n_stocks,
        long_w=cfg.long_weight,
        weight_mode=cfg.weight_mode,
        filter_negative_scores_long=cfg.filter_negative_scores_long
    )

    if nav_df.empty:
        print("[BTR-BT][警告] 未能生成回测净值（可能所有周都空仓或数据缺失）。")
        return

    out_nav_csv = out_dir / f"nav_{cfg.run_name_out}.csv"
    ensure_dir(out_nav_csv)
    cols = ["ret_long", "ret_short", "ret_total", "nav", "n_long"]
    nav_df[cols].to_csv(out_nav_csv, float_format="%.8f")
    print(f"[BTR-BT] 已保存净值序列：{out_nav_csv}")

    out_pos = out_dir / f"positions_{cfg.run_name_out}.csv"
    ensure_dir(out_pos)
    pos_df.to_csv(out_pos, index=False, float_format="%.8f")
    print(f"[BTR-BT] 已保存周度持仓权重：{out_pos}")


if __name__ == "__main__":
    main()