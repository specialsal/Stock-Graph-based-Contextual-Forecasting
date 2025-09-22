# coding: utf-8
"""
btr_metrics.py
基于一个主策略 nav.csv，叠加若干对比 nav.csv，并可指定一个主基准用于超额计算。
输出（写入主策略 nav 同级目录的 metrics_{run_name}/ 子目录）：
- metrics_{run_name}.json（总 + 年度 + 最大回撤起止 + 恢复周期 + 胜率）
- metrics_by_year_{run_name}.csv
- nav_{run_name}.png（主策略 + 对比 + 超额（若有））
- nav_marked_{run_name}.png（带窗口色块）
- 若指定 benchmark：
  - excess_nav_{run_name}_vs_{bench}.csv
  - excess_metrics_{run_name}_vs_{bench}.json
  - excess_metrics_by_year_{run_name}_vs_{bench}.csv

新增：
- 总体与年度的两类胜率：
  - 周度组合加权胜率：win_rate_weekly_weighted
  - 个股层面胜率：win_rate_stockwise
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import pandas as pd
import numpy as np

from config import CFG
from utils import load_calendar, weekly_fridays
from utils_backtest import (
    ensure_dir, read_nav_csv, calc_full_metrics, metrics_by_year_to_df,
    align_nav_frames, excess_nav_from_returns, plot_nav_compare,
    recover_window_spans_from_models
)

# ========= 用户配置（请修改这里） =========
# 主策略 nav.csv 路径（必填）
MAIN_NAV_PATH = r"./backtest_rolling/tr1gat1win50/nav_tr1gat1win50.csv"

# 对比 nav.csv 路径列表（可为空；仅用于绘图对比，不参与“超额”的运算）
COMPARISON_NAV_PATHS = [
    r"./backtest_rolling/others/300_index_nav.csv",
    r"./backtest_rolling/others/500_index_nav.csv",
    r"./backtest_rolling/others/1000_index_nav.csv",
]

# 主基准 nav.csv 路径（可为空；若填写则计算“主策略相对该基准”的超额曲线与超额指标）
# BENCHMARK_NAV_PATH = None
BENCHMARK_NAV_PATH = r"./backtest_rolling/others/300_index_nav.csv"

# 年化频率（默认 52，周频）
ANNUAL_FREQ = 52

# 年化无风险收益率 rf（默认 0.0）
RISK_FREE_ANNUAL = 0.0

# 多条曲线日期对齐方式："intersection"（交集）或 "union"（并集+前向填充）
ALIGN_MODE = "intersection"

# 绘图起止范围（可为空，格式 "YYYY-MM-DD"）
PLOT_START = None  # 例如 "2017-01-01"
PLOT_END   = None  # 例如 "2024-12-31"
# =======================================


def infer_run_name_from_nav_path(nav_path: Path) -> str:
    stem = nav_path.stem
    if stem.startswith("nav_"):
        return stem[len("nav_"):]
    return stem


def to_path_or_none(p: Optional[str]) -> Optional[Path]:
    if p is None or str(p).strip() == "":
        return None
    return Path(p).resolve()


def _read_positions(positions_csv: Path) -> Optional[pd.DataFrame]:
    if positions_csv is None or (not positions_csv.exists()):
        return None
    try:
        pos = pd.read_csv(positions_csv)
        if pos is None or pos.empty:
            return None
        # 需要 date, weight, next_week_ret
        req = {"date", "weight", "next_week_ret"}
        if not req.issubset(set(pos.columns)):
            return None
        pos["date"] = pd.to_datetime(pos["date"], errors="coerce")
        pos = pos.dropna(subset=["date"])
        pos = pos.replace([np.inf, -np.inf], np.nan)
        # 类型统一
        pos["weight"] = pd.to_numeric(pos["weight"], errors="coerce")
        pos["next_week_ret"] = pd.to_numeric(pos["next_week_ret"], errors="coerce")
        return pos
    except Exception:
        return None


def compute_win_rates_from_positions(pos: Optional[pd.DataFrame]) -> Dict[str, float]:
    """
    从 positions DataFrame 计算总体胜率（两类）。
    返回：
      - win_rate_weekly_weighted, n_weeks_for_winrate
      - win_rate_stockwise, n_stocks_for_winrate
    """
    out = {
        "win_rate_weekly_weighted": float("nan"),
        "n_weeks_for_winrate": 0,
        "win_rate_stockwise": float("nan"),
        "n_stocks_for_winrate": 0
    }
    if pos is None or pos.empty:
        return out

    # 个股层面胜率
    pos_valid = pos[pd.notna(pos["next_week_ret"])].copy()
    n_obs = int(pos_valid.shape[0])
    out["n_stocks_for_winrate"] = n_obs
    if n_obs > 0:
        out["win_rate_stockwise"] = float((pos_valid["next_week_ret"] > 0).mean())

    # 周度组合加权胜率（按 date 聚合）：无 apply 警告版本
    sub = pos_valid.dropna(subset=["weight"]).copy()
    if not sub.empty:
        proxy_week_ret = (
            sub.assign(prod=sub["weight"] * sub["next_week_ret"])
               .groupby("date", sort=True)["prod"].sum()
        )
        proxy_week_ret = proxy_week_ret.replace([np.inf, -np.inf], np.nan).dropna()
        n_weeks = int(proxy_week_ret.shape[0])
        out["n_weeks_for_winrate"] = n_weeks
        if n_weeks > 0:
            out["win_rate_weekly_weighted"] = float((proxy_week_ret > 0).mean())
    return out


def compute_yearly_win_rates_from_positions(pos: Optional[pd.DataFrame]) -> Dict[str, Dict[str, float]]:
    """
    逐年胜率（两类），返回 dict: year -> {win_rate_weekly_weighted, n_weeks_for_winrate_year,
                                            win_rate_stockwise, n_stocks_for_winrate_year}
    """
    if pos is None or pos.empty or ("date" not in pos.columns):
        return {}
    out: Dict[str, Dict[str, float]] = {}
    pos_year = pos.copy()
    pos_year["year"] = pos_year["date"].dt.year

    for y, g in pos_year.groupby("year"):
        g = g.replace([np.inf, -np.inf], np.nan)
        # 个股层面（忽略 NaN）
        gv = g[pd.notna(g["next_week_ret"])]
        n_stk = int(gv.shape[0])
        win_stock = float((gv["next_week_ret"] > 0).mean()) if n_stk > 0 else float("nan")

        # 周度组合加权（无 apply 警告版本）
        gv2 = gv.dropna(subset=["weight"]).copy()
        if not gv2.empty:
            proxy_week = (
                gv2.assign(prod=gv2["weight"] * gv2["next_week_ret"])
                   .groupby("date", sort=True)["prod"].sum()
            )
            proxy_week = proxy_week.replace([np.inf, -np.inf], np.nan).dropna()
            n_wk = int(proxy_week.shape[0])
            win_week = float((proxy_week > 0).mean()) if n_wk > 0 else float("nan")
        else:
            n_wk = 0
            win_week = float("nan")

        out[str(int(y))] = {
            "win_rate_weekly_weighted": win_week,
            "n_weeks_for_winrate_year": n_wk,
            "win_rate_stockwise": win_stock,
            "n_stocks_for_winrate_year": n_stk
        }
    return out


def main():
    # 解析路径
    main_path = Path(MAIN_NAV_PATH).resolve()
    comp_paths = [Path(p).resolve() for p in COMPARISON_NAV_PATHS]
    bench_path = to_path_or_none(BENCHMARK_NAV_PATH)

    # 读取 NAV
    df_main = read_nav_csv(main_path)
    dfs = [df_main]
    comp_names: List[str] = []
    for p in comp_paths:
        dfs.append(read_nav_csv(p))
        comp_names.append(infer_run_name_from_nav_path(p))

    # 对齐
    dfs_aligned = align_nav_frames(dfs, how=ALIGN_MODE)
    df_main = dfs_aligned[0]
    comp_aligned = dfs_aligned[1:]
    # 基准单独读取（与主策略交集对齐）
    df_bench = read_nav_csv(bench_path) if bench_path is not None else None

    # 裁剪绘图范围
    if PLOT_START:
        s = pd.to_datetime(PLOT_START)
        df_main = df_main[df_main.index >= s]
        comp_aligned = [d[d.index >= s] for d in comp_aligned]
        if df_bench is not None:
            df_bench = df_bench[df_bench.index >= s]
    if PLOT_END:
        e = pd.to_datetime(PLOT_END)
        df_main = df_main[df_main.index <= e]
        comp_aligned = [d[d.index <= e] for d in comp_aligned]
        if df_bench is not None:
            df_bench = df_bench[df_bench.index <= e]

    if df_main.empty:
        raise RuntimeError("主策略 NAV 序列为空或被裁剪为空，请检查 MAIN_NAV_PATH 与时间范围。")

    # 输出目录：主 nav 同级目录 / metrics_{run_name}
    run_name = infer_run_name_from_nav_path(main_path)
    out_dir = main_path.parent / f"metrics_{run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 计算主策略指标（收益/波动/回撤等）
    total_metrics, metrics_by_year, dd_info = calc_full_metrics(
        df_main, ret_col="ret_total", freq_per_year=ANNUAL_FREQ, rf_annual=RISK_FREE_ANNUAL
    )

    # 胜率：读取 positions 并计算 overall + yearly
    positions_csv = main_path.parent / f"positions_{run_name}.csv"
    pos_df = _read_positions(positions_csv)
    win_overall = compute_win_rates_from_positions(pos_df)
    win_yearly = compute_yearly_win_rates_from_positions(pos_df)

    # 写主 metrics JSON / 年度 CSV
    out_json = out_dir / f"metrics_{run_name}.json"

    # 合并 overall
    overall_payload = dict(total_metrics)
    overall_payload.update({
        "win_rate_weekly_weighted": win_overall.get("win_rate_weekly_weighted", float("nan")),
        "n_weeks_for_winrate": win_overall.get("n_weeks_for_winrate", 0),
        "win_rate_stockwise": win_overall.get("win_rate_stockwise", float("nan")),
        "n_stocks_for_winrate": win_overall.get("n_stocks_for_winrate", 0)
    })

    # 合并 yearly：将胜率信息并入 metrics_by_year 的每一年 dict 中
    metrics_by_year_aug: Dict[str, Dict[str, float]] = {}
    for y, m in metrics_by_year.items():
        add = win_yearly.get(y, {})
        m2 = dict(m)
        m2.update({
            "win_rate_weekly_weighted": add.get("win_rate_weekly_weighted", float("nan")),
            "n_weeks_for_winrate_year": add.get("n_weeks_for_winrate_year", 0),
            "win_rate_stockwise": add.get("win_rate_stockwise", float("nan")),
            "n_stocks_for_winrate_year": add.get("n_stocks_for_winrate_year", 0),
        })
        metrics_by_year_aug[y] = m2

    payload = {
        "overall": overall_payload,
        "max_drawdown": {
            "drawdown": dd_info.drawdown,
            "start_date": None if dd_info.start_date is None else str(dd_info.start_date.date()),
            "end_date": None if dd_info.end_date is None else str(dd_info.end_date.date()),
            "recovery_days": dd_info.recovery_periods  # 此处变量仍叫 recovery_periods，但含义为天数
        },
        "by_year": metrics_by_year_aug
    }
    ensure_dir(out_json)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # 年度 CSV 也包含新增的胜率列
    df_year = metrics_by_year_to_df(metrics_by_year_aug)
    out_year_csv = out_dir / f"metrics_by_year_{run_name}.csv"
    ensure_dir(out_year_csv)
    df_year.to_csv(out_year_csv, index=False)

    # 超额 NAV 与指标（若提供基准）
    excess_series = None
    if df_bench is not None:
        # 与主策略按交集对齐
        [df_main_ex, df_bench_ex] = align_nav_frames([df_main, df_bench], how=ALIGN_MODE)
        ex_nav = excess_nav_from_returns(df_main_ex["ret_total"], df_bench_ex["ret_total"])
        bench_name = infer_run_name_from_nav_path(bench_path)
        ex_name = f"Excess vs {bench_name}"
        excess_series = (ex_name, ex_nav)

        # 导出超额 NAV
        ex_nav_df = pd.DataFrame({"excess_nav": ex_nav.values}, index=ex_nav.index)
        out_ex_nav = out_dir / f"excess_nav_{run_name}_vs_{bench_name}.csv"
        ensure_dir(out_ex_nav)
        ex_nav_df.to_csv(out_ex_nav, float_format="%.8f")

        # 超额指标（基于收益差）
        ex_rets = (df_main_ex["ret_total"].values.astype(float) - df_bench_ex["ret_total"].values.astype(float))
        tmp = pd.DataFrame({"ret_total": ex_rets, "nav": ex_nav.values}, index=ex_nav.index)
        ex_total, ex_by_year, ex_dd = calc_full_metrics(tmp, ret_col="ret_total",
                                                        freq_per_year=ANNUAL_FREQ, rf_annual=RISK_FREE_ANNUAL)
        ex_json = out_dir / f"excess_metrics_{run_name}_vs_{bench_name}.json"
        with open(ex_json, "w", encoding="utf-8") as f:
            json.dump({
                "overall": ex_total,
                "max_drawdown": {
                    "drawdown": ex_dd.drawdown,
                    "start_date": None if ex_dd.start_date is None else str(ex_dd.start_date.date()),
                    "end_date": None if ex_dd.end_date is None else str(ex_dd.end_date.date()),
                    "recovery_days": ex_dd.recovery_periods  # 天数
                },
                "by_year": ex_by_year
            }, f, ensure_ascii=False, indent=2)

        ex_year_csv = out_dir / f"excess_metrics_by_year_{run_name}_vs_{bench_name}.csv"
        metrics_by_year_to_df(ex_by_year).to_csv(ex_year_csv, index=False)

    # 组装绘图数据
    series_dict: Dict[str, pd.Series] = {run_name: df_main["nav"]}
    for p, d in zip(comp_paths, comp_aligned):
        name = infer_run_name_from_nav_path(p)
        if d.empty:
            continue
        series_dict[name] = d["nav"]

    # 绘图（不带色块）
    out_png = out_dir / f"nav_{run_name}.png"
    plot_nav_compare(series_dict, out_png, title=f"NAV Compare [{run_name}]", spans=None, excess_series=excess_series)

    # 绘图（带窗口色块）
    # 使用 CFG.model_dir + CFG.step_weeks 恢复窗口色块，并裁剪到主策略 NAV 的日期范围
    cal = load_calendar(CFG.trading_day_file)
    fridays_all = weekly_fridays(cal)
    bt_start = df_main.index.min()
    bt_end = df_main.index.max()
    spans = recover_window_spans_from_models(
        model_dir=CFG.model_dir,
        fridays_all=fridays_all,
        bt_start=bt_start,
        bt_end=bt_end,
        step_weeks=int(getattr(CFG, "step_weeks", 52))
    )
    out_png_marked = out_dir / f"nav_marked_{run_name}.png"
    plot_nav_compare(series_dict, out_png_marked, title=f"NAV Compare [{run_name}] [windows marked]", spans=spans, excess_series=excess_series)

    print(f"[BTR-METRICS] 已输出至：{out_dir}")


if __name__ == "__main__":
    main()