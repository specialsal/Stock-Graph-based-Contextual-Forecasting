# coding: utf-8
"""
btr_backtest.py
读取 backtest_rolling/{run_name}/predictions_filtered_{run_name}.parquet + 原始日线，
按 Top-K 先选后权重（等权 / 按分数加权），构建周度组合，输出：
- backtest_rolling/{run_name}/positions_{run_name}.csv
- backtest_rolling/{run_name}/nav_{run_name}.csv
图与指标由 btr_metrics.py 统一处理。
"""

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from backtest_rolling_config import BT_ROLL_CFG
from utils import load_calendar, weekly_fridays
from config import CFG

# 可调选项
WEIGHT_MODE = "equal"   # "equal" 或 "score"
FILTER_NEGATIVE_SCORES_LONG = False  # 做多是否过滤负分


def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def compute_weekly_returns(close_pivot: pd.DataFrame,
                           fridays: pd.DatetimeIndex):
    """
    返回：
      - weekly_ret: dict[周五 -> pd.Series(index=stock, value=ret)]
      - start_date_map: dict[周五 -> 实际起始交易日]
    """
    avail = close_pivot.index
    out = {}
    start_date_map = {}
    for i in range(len(fridays) - 1):
        f0 = fridays[i]; f1 = fridays[i+1]
        i0 = avail[avail <= f0]; i1 = avail[avail <= f1]
        if len(i0) == 0 or len(i1) == 0:
            continue
        d0 = i0[-1]; d1 = i1[-1]
        if d0 >= d1:
            continue
        start = close_pivot.loc[d0]
        end   = close_pivot.loc[d1]
        ret = (end / start - 1).replace([np.inf, -np.inf], np.nan)
        out[f0] = ret
        start_date_map[f0] = d0
    return out, start_date_map


def build_portfolio_with_weights(scores: pd.DataFrame,
                                 weekly_ret: Dict[pd.Timestamp, pd.Series],
                                 mode: str,
                                 top_pct: float,
                                 max_n: int,
                                 min_n: int,
                                 long_w: float,
                                 slippage_bps: float,
                                 fee_bps: float,
                                 weight_mode: str = "equal",
                                 filter_negative_scores_long: bool = True,
                                 include_last_no_return_week: bool = True):
    """
    返回：
      - nav_df: index=date, cols=[ret_long, ret_short, ret_total, nav, n_long]
      - pos_df: 每周持仓与权重明细（date, stock, weight, score）
    说明：
      - 本函数实现 long-only（ret_short 固定为 0），保留 ret_short 字段以便扩展。
    """
    if scores.empty:
        return pd.DataFrame(), pd.DataFrame()

    dates_with_ret = sorted(set(scores["date"]).intersection(weekly_ret.keys()))
    recs = []
    pos_rows = []
    cost_long = (slippage_bps + fee_bps) * 1e-4

    # 1) 对能计算收益的周
    for d in dates_with_ret:
        df_d = scores[scores["date"] == d].copy()
        ret_s = weekly_ret[d]
        df_d = df_d.merge(ret_s.rename("ret"), left_on="stock", right_index=True, how="inner")
        df_d = df_d.dropna(subset=["ret", "score"])
        if len(df_d) < min_n:
            recs.append({"date": d, "ret_long": 0.0, "ret_short": 0.0, "ret_total": 0.0, "n_long": 0})
            continue

        df_d = df_d.sort_values(["score","stock"], ascending=[False, True])
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
        total = long_w * (long_ret - cost_long)

        recs.append({"date": d, "ret_long": long_ret, "ret_short": 0.0, "ret_total": total, "n_long": len(df_sel)})
        for _, r in df_sel.iterrows():
            pos_rows.append({"date": d, "stock": r["stock"], "weight": float(r["weight"]), "score": float(r["score"])})

    # 2) 追加仅持仓不计收益的最后一周（若存在）
    if include_last_no_return_week:
        all_score_dates = sorted(set(scores["date"]))
        if all_score_dates:
            last_d = all_score_dates[-1]
            if last_d not in weekly_ret:
                df_last = scores[scores["date"] == last_d].copy()
                df_last = df_last.dropna(subset=["score"])
                if len(df_last) >= min_n:
                    df_last = df_last.sort_values(["score","stock"], ascending=[False, True])
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
                            pos_rows.append({"date": last_d, "stock": r["stock"], "weight": float(r["weight"]), "score": float(r["score"])})

    # 输出
    if not recs:
        return pd.DataFrame(columns=["ret_long","ret_short","ret_total","nav","n_long"]), pd.DataFrame(pos_rows)

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

    # 周度收益
    price_df = pd.read_parquet(cfg.price_day_file)
    if not isinstance(price_df.index, pd.MultiIndex):
        price_df = price_df.set_index(["order_book_id", "date"]).sort_index()
    close_pivot = price_df["close"].unstack(0).sort_index()

    cal = load_calendar(cfg.trading_day_file)
    fridays_all = weekly_fridays(cal)
    weekly_ret_all, _ = compute_weekly_returns(close_pivot, fridays_all)
    weekly_ret = {d: s for d, s in weekly_ret_all.items()
                  if (d >= pd.Timestamp(cfg.bt_start_date)) and (d <= pd.Timestamp(cfg.bt_end_date))}

    # 组合
    nav_df, pos_df = build_portfolio_with_weights(
        scores, weekly_ret,
        mode=cfg.mode,
        top_pct=cfg.top_pct,
        max_n=cfg.max_n_stocks,
        min_n=cfg.min_n_stocks,
        long_w=cfg.long_weight,
        slippage_bps=cfg.slippage_bps,
        fee_bps=cfg.fee_bps,
        weight_mode=WEIGHT_MODE,
        filter_negative_scores_long=FILTER_NEGATIVE_SCORES_LONG
    )

    # 保存
    if nav_df.empty:
        print("[BTR-BT][警告] 未能生成回测净值（可能所有周都空仓或数据缺失）。")
        return

    out_nav_csv = out_dir / f"nav_{cfg.run_name_out}.csv"
    ensure_dir(out_nav_csv)
    # 统一包含 ret_long / ret_short / ret_total / nav / n_long（若未来扩展 LS，可再加 n_short）
    cols = ["ret_long", "ret_short", "ret_total", "nav", "n_long"]
    nav_df[cols].to_csv(out_nav_csv, float_format="%.8f")
    print(f"[BTR-BT] 已保存净值序列：{out_nav_csv}")

    out_pos = out_dir / f"positions_{cfg.run_name_out}.csv"
    ensure_dir(out_pos)
    pos_df.to_csv(out_pos, index=False, float_format="%.8f")
    print(f"[BTR-BT] 已保存周度持仓权重：{out_pos}")


if __name__ == "__main__":
    main()