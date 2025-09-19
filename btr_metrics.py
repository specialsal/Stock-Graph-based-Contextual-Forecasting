# coding: utf-8
"""
btr_metrics.py
基于一个主策略 nav.csv，叠加若干对比 nav.csv，并可指定一个主基准用于超额计算。
输出（写入主策略 nav 同级目录的 metrics_{run_name}/ 子目录）：
- metrics_{run_name}.json（总 + 年度 + 最大回撤起止 + 恢复周期）
- metrics_by_year_{run_name}.csv
- nav_{run_name}.png（主策略 + 对比 + 超额（若有））
- nav_marked_{run_name}.png（带窗口色块）
- 若指定 benchmark：
  - excess_nav_{run_name}_vs_{bench}.csv
  - excess_metrics_{run_name}_vs_{bench}.json
  - excess_metrics_by_year_{run_name}_vs_{bench}.csv

使用方法：
- 直接在 IDE 中打开本文件，修改“用户配置”段的参数，然后运行 main()。
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import pandas as pd

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

    # 计算主策略指标
    total_metrics, metrics_by_year, dd_info = calc_full_metrics(
        df_main, ret_col="ret_total", freq_per_year=ANNUAL_FREQ, rf_annual=RISK_FREE_ANNUAL
    )

    # 写主 metrics JSON / 年度 CSV
    out_json = out_dir / f"metrics_{run_name}.json"
    payload = {
        "overall": total_metrics,
        "max_drawdown": {
            "drawdown": dd_info.drawdown,
            "start_date": None if dd_info.start_date is None else str(dd_info.start_date.date()),
            "end_date": None if dd_info.end_date is None else str(dd_info.end_date.date()),
            "recovery_days": dd_info.recovery_periods  # 此处变量仍叫 recovery_periods，但含义已是天数
        },
        "by_year": metrics_by_year
    }
    ensure_dir(out_json)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    df_year = metrics_by_year_to_df(metrics_by_year)
    out_year_csv = out_dir / f"metrics_by_year_{run_name}.csv"
    ensure_dir(out_year_csv)
    df_year.to_csv(out_year_csv, index=False)

    # 超额 NAV 与指标（若提供基准）
    excess_series = None
    if df_bench is not None:
        # 与主策略按交集对齐
        [df_main_ex, df_bench_ex] = align_nav_frames([df_main, df_bench], how="intersection")
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