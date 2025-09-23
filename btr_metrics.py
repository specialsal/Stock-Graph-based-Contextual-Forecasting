# coding: utf-8
"""
btr_metrics.py
基于主策略与若干对比曲线的“ret_total”交集对齐后，再累乘生成标准化 NAV（起点=1），
避免“先各自生成 NAV 再裁剪”导致的起点不为1问题。
输出（写入主策略 nav 同级目录的 metrics_{run_name}/ 子目录）：
- metrics_{run_name}.json（总 + 年度 + 最大回撤起止 + 恢复周期 + 胜率）
- metrics_by_year_{run_name}.csv
- nav_{run_name}.png（主策略 + 对比 + 超额（若有））
- nav_marked_{run_name}.png（带窗口色块）
- 若指定 benchmark：
  - excess_nav_{run_name}_vs_{bench}.csv
  - excess_metrics_{run_name}_vs_{bench}.json
  - excess_metrics_by_year_{run_name}_vs_{bench}.csv

注意
- 推荐所有输入都含 ret_total；若仅有 nav，则从 nav 反推 ret_total 后参与对齐与计算。
- ALIGN_MODE="intersection" 时，严格按交集对齐 ret_total，再累乘 nav（起点=1）。
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

# ========= 用户配置（请修改这里） =========
cfg = BT_ROLL_CFG

# 主输入（可以是包含 ret_total 的文件；若无 ret_total 但有 nav 也可）
# MAIN_NAV_PATH = f"./backtest_rolling/{cfg.run_name}/s_g_combo/combo_nav.csv"
MAIN_NAV_PATH = f"./backtest_rolling/{cfg.run_name}/nav_{cfg.run_name}.csv"

# 对比曲线（建议提供 ret_total；若只含 nav 也会被反推 ret_total）
COMPARISON_NAV_PATHS = [
    r"./backtest_rolling/others/300_index_nav.csv",
    r"./backtest_rolling/others/500_index_nav.csv",
    r"./backtest_rolling/others/1000_index_nav.csv",
]

# 基准（可为空；用于超额）
BENCHMARK_NAV_PATH = r"./backtest_rolling/others/300_index_nav.csv"
# BENCHMARK_NAV_PATH = None

# 年化频率（周频）
ANNUAL_FREQ = 52

# 年化无风险收益率
RISK_FREE_ANNUAL = 0.0

# 对齐方式："intersection"（交集）或 "union"（并集）
ALIGN_MODE = "intersection"

# 绘图起止范围（可为空，格式 "YYYY-MM-DD"）
PLOT_START = None
PLOT_END   = None
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
        req = {"date", "weight", "next_week_ret"}
        if not req.issubset(set(pos.columns)):
            return None
        pos["date"] = pd.to_datetime(pos["date"], errors="coerce")
        pos = pos.dropna(subset=["date"])
        pos = pos.replace([np.inf, -np.inf], np.nan)
        pos["weight"] = pd.to_numeric(pos["weight"], errors="coerce")
        pos["next_week_ret"] = pd.to_numeric(pos["next_week_ret"], errors="coerce")
        return pos
    except Exception:
        return None


def compute_win_rates_from_positions(pos: Optional[pd.DataFrame]) -> Dict[str, float]:
    out = {
        "win_rate_weekly_weighted": float("nan"),
        "n_weeks_for_winrate": 0,
        "win_rate_stockwise": float("nan"),
        "n_stocks_for_winrate": 0
    }
    if pos is None or pos.empty:
        return out
    pos_valid = pos[pd.notna(pos["next_week_ret"])].copy()
    n_obs = int(pos_valid.shape[0])
    out["n_stocks_for_winrate"] = n_obs
    if n_obs > 0:
        out["win_rate_stockwise"] = float((pos_valid["next_week_ret"] > 0).mean())
    sub = pos_valid.dropna(subset=["weight"]).copy()
    if not sub.empty:
        proxy_week_ret = (
            sub.assign(prod=sub["weight"] * sub["next_week_ret"])
               .groupby("date", sort=True)["prod"].sum()
        ).replace([np.inf, -np.inf], np.nan).dropna()
        n_weeks = int(proxy_week_ret.shape[0])
        out["n_weeks_for_winrate"] = n_weeks
        if n_weeks > 0:
            out["win_rate_weekly_weighted"] = float((proxy_week_ret > 0).mean())
    return out


def compute_yearly_win_rates_from_positions(pos: Optional[pd.DataFrame]) -> Dict[str, Dict[str, float]]:
    if pos is None or pos.empty or ("date" not in pos.columns):
        return {}
    out: Dict[str, Dict[str, float]] = {}
    pos_year = pos.copy()
    pos_year["year"] = pos_year["date"].dt.year
    for y, g in pos_year.groupby("year"):
        g = g.replace([np.inf, -np.inf], np.nan)
        gv = g[pd.notna(g["next_week_ret"])]
        n_stk = int(gv.shape[0])
        win_stock = float((gv["next_week_ret"] > 0).mean()) if n_stk > 0 else float("nan")
        gv2 = gv.dropna(subset=["weight"]).copy()
        if not gv2.empty:
            proxy_week = (
                gv2.assign(prod=gv2["weight"] * gv2["next_week_ret"])
                   .groupby("date", sort=True)["prod"].sum()
            ).replace([np.inf, -np.inf], np.nan).dropna()
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


def read_as_ret(path: Path) -> pd.Series:
    """
    读取单条曲线，返回 ret_total（周收益）的 Series（index=date）。
    优先用 ret_total；若无则从 nav 反推；两者都没有则报错。
    """
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
        # 反推 ret：ret_t = nav_t / nav_{t-1} - 1
        ret = nav.pct_change().dropna()
        return ret

    raise RuntimeError(f"文件缺少 ret_total 或 nav 列：{path}")


def align_returns(series_list: List[pd.Series], how: str = "intersection") -> List[pd.Series]:
    """
    对多条 ret_total 做对齐。
    - intersection：索引交集
    - union：索引并集，缺失填 0（仅用于可视化对齐；指标含填充期）
    返回与输入等长的列表。
    """
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
    """
    用 ret 累乘生成 nav，并将第一期明确设为 1.0（保证起点一致）。
    要求 ret 已按目标索引对齐（如交集）。
    """
    nav = (1.0 + pd.Series(ret, copy=True)).cumprod()
    if len(nav) > 0:
        nav.iloc[0:1] = 1.0
    return nav


def main(MAIN_NAV_PATH):
    # 解析路径
    main_path = Path(MAIN_NAV_PATH).resolve()
    comp_paths = [Path(p).resolve() for p in COMPARISON_NAV_PATHS]
    bench_path = to_path_or_none(BENCHMARK_NAV_PATH)

    # 读取为 ret
    ret_main = read_as_ret(main_path)
    ret_comp = [read_as_ret(p) for p in comp_paths]
    ret_bench = read_as_ret(bench_path) if bench_path is not None else None

    # 对齐（核心：先对 ret 取交集，再生成 nav）
    rets_all = [ret_main] + ret_comp
    rets_aligned = align_returns(rets_all, how=ALIGN_MODE)
    ret_main_aligned = rets_aligned[0]
    ret_comp_aligned = rets_aligned[1:]

    # 基准与主策略单独对齐（用于超额/指标）
    if ret_bench is not None:
        [ret_main_ex, ret_bench_ex] = align_returns([ret_main, ret_bench], how=ALIGN_MODE)
    else:
        ret_main_ex, ret_bench_ex = ret_main_aligned, None

    # 裁剪绘图范围（对 ret 裁剪，再生成 nav）
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
        raise RuntimeError("主策略序列为空或被裁剪为空，请检查 MAIN_NAV_PATH 与时间范围。")

    # 输出目录
    run_name = infer_run_name_from_nav_path(main_path)
    out_dir = main_path.parent / f"metrics_{run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 用“交集后的 ret”生成标准化 nav（起点=1）
    nav_main = ret_to_nav_standardized(ret_main_plot)
    nav_comp = [ret_to_nav_standardized(r) for r in ret_comp_plot]

    # 指标（基于 ret_main_ex）
    nav_for_metrics = ret_to_nav_standardized(ret_main_ex)
    main_metrics_df = pd.DataFrame({"ret_total": ret_main_ex, "nav": nav_for_metrics})
    total_metrics, metrics_by_year, dd_info = calc_full_metrics(
        main_metrics_df, ret_col="ret_total", freq_per_year=ANNUAL_FREQ, rf_annual=RISK_FREE_ANNUAL
    )

    # 胜率
    positions_csv = main_path.parent / f"positions_{run_name}.csv"
    pos_df = _read_positions(positions_csv)
    win_overall = compute_win_rates_from_positions(pos_df)
    win_yearly = compute_yearly_win_rates_from_positions(pos_df)

    # 写主 metrics JSON / 年度 CSV
    out_json = out_dir / f"metrics_{run_name}.json"
    overall_payload = dict(total_metrics)
    overall_payload.update({
        "win_rate_weekly_weighted": win_overall.get("win_rate_weekly_weighted", float("nan")),
        "n_weeks_for_winrate": win_overall.get("n_weeks_for_winrate", 0),
        "win_rate_stockwise": win_overall.get("win_rate_stockwise", float("nan")),
        "n_stocks_for_winrate": win_overall.get("n_stocks_for_winrate", 0)
    })
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
            "recovery_days": dd_info.recovery_periods
        },
        "by_year": metrics_by_year_aug
    }
    ensure_dir(out_json)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # 年度 CSV
    df_year = metrics_by_year_to_df(metrics_by_year_aug)
    out_year_csv = out_dir / f"metrics_by_year_{run_name}.csv"
    ensure_dir(out_year_csv)
    df_year.to_csv(out_year_csv, index=False)

    # 超额 NAV 与指标（若提供基准）
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

    # 组装绘图数据（全部由“交集后的 ret 累乘得到的 nav”，起点=1）
    series_dict: Dict[str, pd.Series] = {run_name: nav_main}
    for p, r_nav in zip(comp_paths, nav_comp):
        name = infer_run_name_from_nav_path(p)
        if r_nav.empty:
            continue
        series_dict[name] = r_nav

    # 绘图（不带色块）
    out_png = out_dir / f"nav_{run_name}.png"
    plot_nav_compare(series_dict, out_png, title=f"NAV Compare [{run_name}]", spans=None, excess_series=excess_series)

    # 绘图（带窗口色块）
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