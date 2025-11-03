# coding: utf-8
"""
btr_metrics.py（日度版）
读取“日度”主策略 nav 文件，基于 ret_total 做交集对齐后生成标准化 NAV（起点=1），输出：
- metrics_{run_name}.json（总 + 年度 + 最大回撤起止 + 恢复周期）
- metrics_by_year_{run_name}.csv
- nav_{run_name}.png（主策略 + 对比 + 超额）
- nav_marked_{run_name}.png（带窗口色块）
注意：
- 默认年化频率 = 252（日度）
- 已移除以下统计项的计算与输出：
  n_periods, win_rate_weekly_weighted, n_weeks_for_winrate, win_rate_stockwise, n_stocks_for_winrate
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
    excess_nav_from_returns, plot_nav_compare,
    recover_window_spans_from_models
)
from backtest_rolling_config import BT_ROLL_CFG

# ========= 配置 =========
cfg = BT_ROLL_CFG

MAIN_NAV_PATH = f"./backtest_rolling/{cfg.run_name}/nav_{cfg.run_name}.csv"

COMPARISON_NAV_PATHS = [
    r"./backtest_rolling/others/300_index_nav.csv",
    r"./backtest_rolling/others/500_index_nav.csv",
    r"./backtest_rolling/others/1000_index_nav.csv",
]

BENCHMARK_NAV_PATH = r"./backtest_rolling/others/300_index_nav.csv"  # 可设为 None

ANNUAL_FREQ = 252  # 日度
RISK_FREE_ANNUAL = 0.0
ALIGN_MODE = "intersection"
PLOT_START = None
PLOT_END   = None
# =======================


def infer_run_name_from_nav_path(nav_path: Path) -> str:
    stem = nav_path.stem
    if stem.startswith("nav_"):
        return stem[len("nav_"):]
    return stem


def to_path_or_none(p: Optional[str]) -> Optional[Path]:
    if p is None or str(p).strip() == "":
        return None
    return Path(p).resolve()


def read_as_ret(path: Path) -> pd.Series:
    df = read_nav_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date")
    df = df.replace([np.inf, -np.inf], np.nan).sort_index()
    if "ret_total" in df.columns and df["ret_total"].notna().any():
        ret = pd.to_numeric(df["ret_total"], errors="coerce").dropna()
        return ret
    if "nav" in df.columns and df["nav"].notna().any():
        nav = pd.to_numeric(df["nav"], errors="coerce").dropna()
        ret = nav.pct_change().dropna()
        return ret
    raise RuntimeError(f"文件缺少 ret_total 或 nav 列：{path}")


def align_returns(series_list: List[pd.Series], how: str = "intersection") -> List[pd.Series]:
    how = how.lower()
    if how not in ("intersection", "union"):
        how = "intersection"
    if how == "intersection":
        idx = None
        for s in series_list:
            idx = s.index if idx is None else idx.intersection(s.index)
        idx = idx.sort_values()
        return [s.reindex(idx).dropna() for s in series_list]
    else:
        idx = None
        for s in series_list:
            idx = s.index if idx is None else idx.union(s.index)
        idx = idx.sort_values()
        return [s.reindex(idx).fillna(0.0) for s in series_list]


def ret_to_nav_standardized(ret: pd.Series) -> pd.Series:
    nav = (1.0 + pd.Series(ret, copy=True)).cumprod()
    if len(nav) > 0:
        nav.iloc[0:1] = 1.0
    return nav


def main(MAIN_NAV_PATH):
    main_path = Path(MAIN_NAV_PATH).resolve()
    comp_paths = [Path(p).resolve() for p in COMPARISON_NAV_PATHS]
    bench_path = to_path_or_none(BENCHMARK_NAV_PATH)

    # 读取与对齐（按交集）
    ret_main = read_as_ret(main_path)
    ret_comp = [read_as_ret(p) for p in comp_paths]
    ret_bench = read_as_ret(bench_path) if bench_path is not None else None

    rets_all = [ret_main] + ret_comp
    rets_aligned = align_returns(rets_all, how=ALIGN_MODE)
    ret_main_aligned = rets_aligned[0]
    ret_comp_aligned = rets_aligned[1:]

    if ret_bench is not None:
        [ret_main_ex, ret_bench_ex] = align_returns([ret_main, ret_bench], how=ALIGN_MODE)
    else:
        ret_main_ex, ret_bench_ex = ret_main_aligned, None

    # 裁剪绘图范围
    def clip_ret(r: pd.Series) -> pd.Series:
        if PLOT_START:
            r = r[r.index >= pd.to_datetime(PLOT_START)]
        if PLOT_END:
            r = r[r.index <= pd.to_datetime(PLOT_END)]
        return r

    ret_main_plot = clip_ret(ret_main_aligned)
    ret_comp_plot = [clip_ret(r) for r in ret_comp_aligned]
    if ret_bench_ex is not None:
        ret_main_ex = clip_ret(ret_main_ex)
        ret_bench_ex = clip_ret(ret_bench_ex)

    if ret_main_plot.empty:
        raise RuntimeError("主策略序列为空或被裁剪为空。")

    # 输出目录
    run_name = infer_run_name_from_nav_path(main_path)
    out_dir = main_path.parent / f"metrics_{run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 用对齐后的 ret 生成标准化 nav
    nav_main = ret_to_nav_standardized(ret_main_plot)
    nav_comp = [ret_to_nav_standardized(r) for r in ret_comp_plot]

    # 指标（基于日度；calc_full_metrics 里会用 freq_per_year=252）
    nav_for_metrics = ret_to_nav_standardized(ret_main_ex)
    main_metrics_df = pd.DataFrame({"ret_total": ret_main_ex, "nav": nav_for_metrics})
    total_metrics, metrics_by_year, dd_info = calc_full_metrics(
        main_metrics_df, ret_col="ret_total", freq_per_year=ANNUAL_FREQ, rf_annual=RISK_FREE_ANNUAL
    )

    # 写主 metrics JSON / 年度 CSV（已移除指定统计项）
    out_json = out_dir / f"metrics_{run_name}.json"
    overall_payload = dict(total_metrics)  # 不再添加被要求删除的指标

    payload = {
        "overall": overall_payload,
        "max_drawdown": {
            "drawdown": dd_info.drawdown,
            "start_date": None if dd_info.start_date is None else str(dd_info.start_date.date()),
            "end_date": None if dd_info.end_date is None else str(dd_info.end_date.date()),
            "recovery_days": dd_info.recovery_periods
        },
        "by_year": metrics_by_year
    }
    ensure_dir(out_json)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # 年度 CSV
    df_year = metrics_by_year_to_df(metrics_by_year)
    out_year_csv = out_dir / f"metrics_by_year_{run_name}.csv"
    ensure_dir(out_year_csv)
    df_year.to_csv(out_year_csv, index=False)

    # 超额 NAV 与指标（若提供基准，仅作为图表展示与附加输出）
    excess_series = None
    if ret_bench_ex is not None and (not ret_bench_ex.empty):
        ex_nav = excess_nav_from_returns(ret_main_ex, ret_bench_ex)
        bench_name = infer_run_name_from_nav_path(bench_path)
        ex_name = f"Excess vs {bench_name}"
        excess_series = (ex_name, ex_nav)

        # 导出超额 NAV
        ex_nav_df = pd.DataFrame({"excess_nav": ex_nav.values}, index=ex_nav.index)
        out_ex_nav = out_dir / f"excess_nav_{run_name}_vs_{bench_name}.csv"
        ensure_dir(out_ex_nav)
        ex_nav_df.to_csv(out_ex_nav, float_format="%.8f")

        # 超额指标（用收益差）
        ex_rets = (ret_main_ex.values.astype(float) - ret_bench_ex.values.astype(float))
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
                    "recovery_days": ex_dd.recovery_periods
                },
                "by_year": ex_by_year
            }, f, ensure_ascii=False, indent=2)

        ex_year_csv = out_dir / f"excess_metrics_by_year_{run_name}_vs_{bench_name}.csv"
        metrics_by_year_to_df(ex_by_year).to_csv(ex_year_csv, index=False)

    # 组装绘图数据（全用交集后的 ret 累乘得到 nav，起点=1）
    series_dict: Dict[str, pd.Series] = {run_name: nav_main}
    for p, r_nav in zip(comp_paths, nav_comp):
        name = infer_run_name_from_nav_path(p)
        if r_nav.empty:
            continue
        series_dict[name] = r_nav

    # 绘图（不带色块）
    out_png = out_dir / f"nav_{run_name}.png"
    plot_nav_compare(series_dict, out_png, title=f"NAV Compare [{run_name}]", spans=None, excess_series=excess_series)

    # 窗口色块（仍按训练窗口推断）
    cal = load_calendar(CFG.trading_day_file)
    fridays_all = weekly_fridays(cal)
    bt_start = nav_main.index.min()
    bt_end = nav_main.index.max()
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
    main(MAIN_NAV_PATH)