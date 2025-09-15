# coding: utf-8
"""
btr_backtest.py
滚动回测（组合阶段拆分版）：读取池内打分 parquet + 原始日线，
按 Top-K 先选后权重（等权 / 按分数加权），构建周度组合并输出指标/净值/持仓权重明细。

输出：
- backtest_rolling/{run_name}/metrics_{run_name_out}.json
- backtest_rolling/{run_name}/nav_{run_name_out}.csv
- backtest_rolling/{run_name}/nav_{run_name_out}.png
- backtest_rolling/{run_name}/nav_marked_{run_name_out}.png
- backtest_rolling/{run_name}/metrics_by_window_{run_name_out}.csv
- backtest_rolling/{run_name}/positions_{run_name_out}.csv

可选参数（在本脚本顶部配置）：
- WEIGHT_MODE: "equal" 或 "score"（线性，做多时可选过滤负分）
- FILTER_NEGATIVE_SCORES_LONG: True/False（做多是否过滤负分；默认 True）

说明：
- 先按原逻辑筛选 Top-K（top_pct + max_n_stocks，min_n_stocks 兜底），然后在这 K 只内做权重分配并归一化到 1；
- 若当周可交易股票数量 < min_n_stocks，则本周空仓；
- 成本沿用 backtest_rolling_config.py（slippage_bps/fee_bps），周度调仓扣一次。
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from backtest_rolling_config import BT_ROLL_CFG
from utils import load_calendar, weekly_fridays
from config import CFG

plt.switch_backend("Agg")

# ===== 用户可调的新增选项 =====
WEIGHT_MODE = "score"   # "equal" 或 "score"
FILTER_NEGATIVE_SCORES_LONG = True  # 做多时过滤负分

# ===== 工具函数 =====
def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def compute_weekly_returns(close_pivot: pd.DataFrame,
                           fridays: pd.DatetimeIndex):
    """
    将日线收盘价转换为周度收益（上周五到下周五最近交易日收盘的收益）。
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

def save_nav_plot(nav_df: pd.DataFrame, out_png: Path, title: str):
    ensure_dir(out_png)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(nav_df.index, nav_df["nav"], label="Strategy NAV", color="tab:blue")
    ax.set_title(title); ax.set_xlabel("Date"); ax.set_ylabel("NAV")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)

def save_nav_plot_marked(nav_df: pd.DataFrame,
                         window_spans: List[Tuple[pd.Timestamp, pd.Timestamp]],
                         out_png: Path,
                         title: str):
    ensure_dir(out_png)
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.plot(nav_df.index, nav_df["nav"], color="tab:blue", label="Strategy NAV", zorder=2)
    colors = ["#FFEDA0", "#AEDFF7", "#C7F2C8", "#FBC4AB", "#D9D7F1", "#D7F2BA", "#FDE68A", "#CFE8F3"]
    for i, (s, e) in enumerate(window_spans):
        c = colors[i % len(colors)]
        ax.axvspan(s, e, color=c, alpha=0.25, zorder=1)
    ax.set_title(title); ax.set_xlabel("Date"); ax.set_ylabel("NAV")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)

def calc_stats(nav_df: pd.DataFrame, freq_per_year: int = 52) -> Dict[str, float]:
    if nav_df.empty:
        return {}
    nav = nav_df["nav"].values
    rets = nav_df["ret_total"].values
    total_return = float(nav[-1] - 1.0)
    ann_return = float((1.0 + rets).prod() ** (freq_per_year / max(1, len(rets))) - 1.0)
    vol = float(np.std(rets, ddof=1)) * math.sqrt(freq_per_year) if len(rets) > 1 else 0.0
    sharpe = float(ann_return / vol) if vol > 1e-12 else float("nan")
    peak = -np.inf; max_dd = 0.0
    for v in nav:
        peak = max(peak, v) if np.isfinite(peak) else v
        dd = (v / peak - 1.0) if peak > 0 else 0.0
        max_dd = min(max_dd, dd)
    max_drawdown = float(-max_dd)
    calmar = float(ann_return / max_drawdown) if max_drawdown > 1e-12 else float("nan")
    return {
        "total_return": total_return,
        "annual_return": ann_return,
        "annual_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "n_periods": int(len(rets))
    }

def recover_window_spans(model_dir: Path,
                         fridays_all: pd.DatetimeIndex,
                         bt_start: pd.Timestamp,
                         bt_end: pd.Timestamp,
                         step_weeks: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    根据模型目录里的 model_best_YYYYMMDD.pth 序列，结合 step_weeks 恢复每个窗口在回测区间内的跨度。
    逻辑：
      - 取每个 model_best_* 的日期作为窗口起点（对齐到最近不超过它的周五）
      - 窗口长度 step_weeks，终点为起点向后 step_weeks-1 个周五（或到达序列末尾/回测结束）
    """
    best_files = sorted(model_dir.glob("model_best_*.pth"))
    start_dates = []
    for f in best_files:
        try:
            tag = f.stem.split("_")[-1]
            dt = pd.to_datetime(tag)
            # 对齐到不超过 dt 的最近周五
            if dt not in fridays_all:
                ok = fridays_all[fridays_all <= dt]
                if len(ok) == 0:
                    continue
                dt = ok[-1]
            start_dates.append(dt)
        except Exception:
            continue
    start_dates = sorted(set(start_dates))
    if not start_dates:
        return []

    spans = []
    for s in start_dates:
        idx = fridays_all.get_indexer([s])[0]
        j = min(idx + step_weeks - 1, len(fridays_all) - 1)
        e = fridays_all[j]
        # 裁剪到回测区间
        ss = max(s, bt_start)
        ee = min(e, bt_end)
        if ss <= ee:
            spans.append((ss, ee))

    # 合并重叠/相邻的 span，避免色块过密
    spans_sorted = sorted(spans, key=lambda x: x[0])
    merged: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for s, e in spans_sorted:
        if not merged:
            merged.append([s, e])
        else:
            ps, pe = merged[-1]
            if s <= pe:  # 重叠或相接
                merged[-1][1] = max(pe, e)
            else:
                merged.append([s, e])
    return [(s, e) for s, e in merged]

# ===== 组合构建（long-only，支持等权/按分数加权）=====
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
      - nav_df: index=date, cols=[ret_total, nav, n_long]
      - pos_df: 每周持仓与权重明细（date, stock, weight, score）
    注意：
      - 做多模式，若 filter_negative_scores_long=True，则剔除 score<=0 的标的；
      - 当周可交易股票数量 < min_n 时，空仓（收益=0）；
      - 若 include_last_no_return_week=True，则对“scores 中存在但 weekly_ret 中不存在的最后一个周五”
        也输出持仓明细（仅 pos，不计入 nav/收益）。
    """
    if scores.empty:
        return pd.DataFrame(), pd.DataFrame()

    # 参与收益计算的日期（有下周收益的周五）
    dates_with_ret = sorted(set(scores["date"]).intersection(weekly_ret.keys()))
    recs = []
    pos_rows = []
    cost_long = (slippage_bps + fee_bps) * 1e-4

    # 1) 正常回测：仅对 dates_with_ret 计算收益与持仓
    for d in dates_with_ret:
        df_d = scores[scores["date"] == d].copy()
        ret_s = weekly_ret[d]
        df_d = df_d.merge(ret_s.rename("ret"), left_on="stock", right_index=True, how="inner")
        df_d = df_d.dropna(subset=["ret", "score"])
        if len(df_d) < min_n:
            recs.append({"date": d, "ret_total": 0.0, "n_long": 0})
            continue

        # 排序 + Top-K
        df_d = df_d.sort_values(["score","stock"], ascending=[False, True])
        n = len(df_d)
        n_target = max(0, min(int(math.floor(n * top_pct)), max_n))
        if n_target == 0 and n >= min_n:
            n_target = min(min_n, n)
        df_sel = df_d.head(n_target).copy()

        # 做多时可选过滤负分
        if filter_negative_scores_long:
            df_sel = df_sel[df_sel["score"] > 0]

        if len(df_sel) < min_n:
            recs.append({"date": d, "ret_total": 0.0, "n_long": 0})
            continue

        # 权重
        if weight_mode == "equal":
            w = np.full(len(df_sel), 1.0 / len(df_sel), dtype=float)
        else:
            s = df_sel["score"].values.astype(float)
            s = np.clip(s, 0.0, None)
            ssum = s.sum()
            if ssum <= 0:
                w = np.full(len(df_sel), 1.0 / len(df_sel), dtype=float)
            else:
                w = s / ssum

        df_sel = df_sel.assign(weight=w)

        # 组合周收益
        long_ret = float((df_sel["ret"] * df_sel["weight"]).sum())
        total = long_w * (long_ret - cost_long)

        recs.append({"date": d, "ret_total": total, "n_long": len(df_sel)})
        for _, r in df_sel.iterrows():
            pos_rows.append({"date": d, "stock": r["stock"], "weight": float(r["weight"]), "score": float(r["score"])})

    # 2) 追加“仅持仓、不计收益”的最后一个周五
    if include_last_no_return_week:
        all_score_dates = sorted(set(scores["date"]))
        if all_score_dates:
            last_d = all_score_dates[-1]
            if last_d not in weekly_ret:
                # 对该周做与上面相同的 Top-K 与权重（但不合并周收益）
                df_last = scores[scores["date"] == last_d].copy()
                # 此处没有 ret 列（没有下周收益），所以仅用 score 进行筛选与加权
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
                            if ssum <= 0:
                                w = np.full(len(df_sel), 1.0 / len(df_sel), dtype=float)
                            else:
                                w = s / ssum
                        df_sel = df_sel.assign(weight=w)
                        for _, r in df_sel.iterrows():
                            pos_rows.append({"date": last_d, "stock": r["stock"], "weight": float(r["weight"]), "score": float(r["score"])})

    # 3) 生成 nav_df 与 pos_df
    if not recs:
        # 没有任何可计算收益的周；仍可返回仅 positions（若存在 last_d）
        nav_df = pd.DataFrame(columns=["ret_total", "nav", "n_long"])
        pos_df = pd.DataFrame(pos_rows)
        if not pos_df.empty:
            pos_df = pos_df.sort_values(["date", "weight"], ascending=[True, False])
        return nav_df, pos_df

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
    # 过滤到回测区间
    scores = scores[(scores["date"] >= pd.Timestamp(cfg.bt_start_date)) & (scores["date"] <= pd.Timestamp(cfg.bt_end_date))]
    if scores.empty:
        raise RuntimeError("回测区间内打分为空")

    # 计算周度收益
    price_df = pd.read_parquet(cfg.price_day_file)
    if not isinstance(price_df.index, pd.MultiIndex):
        price_df = price_df.set_index(["order_book_id", "date"]).sort_index()
    close_pivot = price_df["close"].unstack(0).sort_index()

    cal = load_calendar(cfg.trading_day_file)
    fridays_all = weekly_fridays(cal)
    weekly_ret_all, _ = compute_weekly_returns(close_pivot, fridays_all)
    weekly_ret = {d: s for d, s in weekly_ret_all.items()
                  if (d >= pd.Timestamp(cfg.bt_start_date)) and (d <= pd.Timestamp(cfg.bt_end_date))}

    # 恢复窗口跨度，用于标注背景色
    model_dir = Path(cfg.model_dir)
    spans_clipped = recover_window_spans(
        model_dir=model_dir,
        fridays_all=fridays_all,
        bt_start=pd.Timestamp(cfg.bt_start_date),
        bt_end=pd.Timestamp(cfg.bt_end_date),
        step_weeks=int(getattr(CFG, "step_weeks", 4))
    )

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

    if nav_df.empty:
        print("[BTR-BT][警告] 未能生成回测净值（可能所有周都空仓或数据缺失）。")
        return

    # 输出净值与图表
    out_nav_csv = out_dir / f"nav_{cfg.run_name_out}.csv"
    ensure_dir(out_nav_csv)
    nav_df[["ret_total", "nav", "n_long"]].to_csv(out_nav_csv, float_format="%.8f")
    print(f"[BTR-BT] 已保存净值序列：{out_nav_csv}")

    out_png = out_dir / f"nav_{cfg.run_name_out}.png"
    title = f"Backtest [{cfg.mode}] {WEIGHT_MODE}-weighted (step={getattr(CFG, 'step_weeks', 4)})"
    save_nav_plot(nav_df, out_png, title=title)
    print(f"[BTR-BT] 已保存净值图：{out_png}")

    out_png_marked = out_dir / f"nav_marked_{cfg.run_name_out}.png"
    save_nav_plot_marked(nav_df, spans_clipped, out_png_marked, title=title + " [windows marked]")
    print(f"[BTR-BT] 已保存标注窗口净值图：{out_png_marked}")

    # 指标（总）
    metrics = calc_stats(nav_df)
    out_json = out_dir / f"metrics_{cfg.run_name_out}.json"
    ensure_dir(out_json)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("[BTR-BT] 指标汇总：")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    # 保存每周持仓与权重明细
    out_pos = out_dir / f"positions_{cfg.run_name_out}.csv"
    ensure_dir(out_pos)
    pos_df.to_csv(out_pos, index=False, float_format="%.8f")
    print(f"[BTR-BT] 已保存周度持仓权重：{out_pos}")

    # 窗口级指标（占位：需要更精确的窗口边界可在此按 spans_clipped 切分计算）
    out_win = out_dir / f"metrics_by_window_{cfg.run_name_out}.csv"
    ensure_dir(out_win)
    pd.DataFrame([]).to_csv(out_win, index=False)
    print(f"[BTR-BT] 已保存窗口级指标（空占位）：{out_win}")

if __name__ == "__main__":
    main()