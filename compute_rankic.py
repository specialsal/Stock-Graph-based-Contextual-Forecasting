# coding: utf-8
"""
compute_rankic.py
计算与可视化周度 RankIC（Spearman）序列，并按年份输出统计：
- 信号锚点：scores['date']（周五）
- 收益口径：O2C（下一交易日开盘 -> 下个周五收盘）
- 成本口径：默认单股价格层乘法计入（非对称手续费 + 对称滑点，读取 backtest_rolling_config.py）

输出：
- backtest_rolling/{run_name}/rankic_{run_name}.csv
- backtest_rolling/{run_name}/rankic_timeseries_{run_name}.png
- backtest_rolling/{run_name}/rankic_hist_{run_name}.png
- backtest_rolling/{run_name}/yearly_rankic_stats_{run_name}.csv
- backtest_rolling/{run_name}/rankic_yearly_mean_{run_name}.png
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

from backtest_rolling_config import BT_ROLL_CFG
from utils import load_calendar, weekly_fridays


def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def _cost_terms_from_cfg(cfg):
    # 返回买/卖侧单边综合成本（滑点+手续费），单位小数
    slp = float(getattr(cfg, "slippage_bps", 0.0)) * 1e-4
    buy_fee = float(getattr(cfg, "buy_fee_bps", 0.0)) * 1e-4
    sell_fee = float(getattr(cfg, "sell_fee_bps", 0.0)) * 1e-4
    return slp + buy_fee, slp + sell_fee  # buy_cost, sell_cost


def compute_o2c_returns_for_rankic(close_pivot: pd.DataFrame,
                                   open_pivot: pd.DataFrame,
                                   fridays: pd.DatetimeIndex,
                                   use_price_cost: bool,
                                   buy_cost: float,
                                   sell_cost: float) -> Tuple[Dict[pd.Timestamp, pd.Series], Dict[pd.Timestamp, pd.Timestamp]]:
    """
    计算 O2C 单股收益（可在价格层计入成本）
    返回：
      - weekly_ret: dict[周五锚点 -> Series(index=stock, value=ret)]
      - start_date_map: dict[周五锚点 -> 实际起始交易日]
    """
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
            C1 = close_pivot.loc[d1]
        except KeyError:
            continue

        if use_price_cost:
            O0_eff = O0 * (1.0 + buy_cost)
            C1_eff = C1 * (1.0 - sell_cost)
            ret = C1_eff / O0_eff - 1.0
        else:
            ret = C1 / O0 - 1.0

        ret = ret.replace([np.inf, -np.inf], np.nan)
        out[f0] = ret
        start_date_map[f0] = d_start

    return out, start_date_map


def plot_rankic_timeseries(ic_df: pd.DataFrame, out_path: Path, title: str = "Weekly RankIC"):
    plt.figure(figsize=(10, 4.5), dpi=160)
    x = ic_df.index
    y = ic_df["rank_ic"].values
    mean_val = np.nanmean(y)

    plt.plot(x, y, color="#2F6CFF", linewidth=1.4, label="RankIC")
    plt.axhline(0.0, color="#999999", linewidth=1.0, linestyle="-")
    plt.axhline(mean_val, color="#FF7F0E", linewidth=1.2, linestyle="--", label=f"Mean = {mean_val:.3f}")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("RankIC")
    plt.legend(loc="best", frameon=False)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    ensure_dir(out_path)
    plt.savefig(out_path)
    plt.close()


def plot_rankic_hist(ic_df: pd.DataFrame, out_path: Path, title: str = "RankIC Distribution"):
    plt.figure(figsize=(6.5, 4.5), dpi=160)
    y = ic_df["rank_ic"].dropna().values
    mean_val = np.nanmean(y)

    plt.hist(y, bins=30, color="#2F6CFF", alpha=0.75, edgecolor="white")
    plt.axvline(0.0, color="#999999", linewidth=1.0)
    plt.axvline(mean_val, color="#FF7F0E", linewidth=1.2, linestyle="--", label=f"Mean = {mean_val:.3f}")
    plt.title(title)
    plt.xlabel("RankIC")
    plt.ylabel("Frequency")
    plt.legend(loc="best", frameon=False)
    plt.grid(True, linestyle="--", alpha=0.25)
    plt.tight_layout()
    ensure_dir(out_path)
    plt.savefig(out_path)
    plt.close()


def plot_yearly_mean_bar(stats_df: pd.DataFrame, out_path: Path, title: str = "Yearly RankIC Mean"):
    plt.figure(figsize=(8.5, 4.5), dpi=160)
    years = stats_df.index.astype(str).tolist()
    means = stats_df["ic_mean"].values
    overall_mean = stats_df.attrs.get("overall_mean", np.nan)

    bars = plt.bar(years, means, color="#2F6CFF", alpha=0.85, edgecolor="white")
    plt.axhline(0.0, color="#999999", linewidth=1.0)
    if np.isfinite(overall_mean):
        plt.axhline(overall_mean, color="#FF7F0E", linewidth=1.2, linestyle="--", label=f"Overall Mean = {overall_mean:.3f}")
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("RankIC Mean")
    # 在柱子上方标注数值
    for b, v in zip(bars, means):
        plt.text(b.get_x() + b.get_width()/2, b.get_height() + (0.002 if v>=0 else -0.002),
                 f"{v:.3f}", ha="center", va="bottom" if v>=0 else "top", fontsize=9, color="#333333")
    plt.legend(loc="best", frameon=False)
    plt.grid(True, axis="y", linestyle="--", alpha=0.25)
    plt.tight_layout()
    ensure_dir(out_path)
    plt.savefig(out_path)
    plt.close()


def main():
    cfg = BT_ROLL_CFG
    out_dir = cfg.backtest_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 读取预测
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

    # 日线数据
    price_df = pd.read_parquet(cfg.price_day_file)
    if not isinstance(price_df.index, pd.MultiIndex):
        price_df = price_df.set_index(["order_book_id", "date"]).sort_index()
    need_cols = ["open", "close"]
    lack = [c for c in need_cols if c not in price_df.columns]
    if lack:
        raise RuntimeError(f"price_day_file 缺少必要列：{lack}")

    close_pivot = price_df["close"].unstack(0).sort_index()
    open_pivot  = price_df["open"].unstack(0).sort_index()

    # 周锚
    cal = load_calendar(cfg.trading_day_file)
    fridays_all = weekly_fridays(cal)

    # 成本口径
    use_price_cost = True  # 与当前回测一致；如需对比旧口径可改为 False
    buy_cost, sell_cost = _cost_terms_from_cfg(cfg)

    # 计算 O2C 收益
    weekly_ret_all, _ = compute_o2c_returns_for_rankic(
        close_pivot=close_pivot,
        open_pivot=open_pivot,
        fridays=fridays_all,
        use_price_cost=use_price_cost,
        buy_cost=buy_cost,
        sell_cost=sell_cost
    )

    # 仅保留有打分且有收益的周
    dates = sorted(set(scores["date"]).intersection(weekly_ret_all.keys()))
    if not dates:
        raise RuntimeError("没有可计算 RankIC 的重叠周")

    # 计算 RankIC
    recs = []
    for d in dates:
        df_d = scores[scores["date"] == d][["stock", "score"]].dropna()
        ret_s = weekly_ret_all[d]
        s = df_d.merge(ret_s.rename("ret"), left_on="stock", right_index=True, how="inner").dropna(subset=["score", "ret"])
        if len(s) < 5:
            continue
        ic_val, _ = spearmanr(s["score"].values, s["ret"].values)
        recs.append({"date": d, "rank_ic": float(ic_val)})

    if not recs:
        raise RuntimeError("RankIC 结果为空（样本过少或数据缺失）")

    ic_df = pd.DataFrame(recs).sort_values("date").set_index("date")

    # 整体统计
    ic_mean = ic_df["rank_ic"].mean()
    ic_std = ic_df["rank_ic"].std(ddof=1)
    icir = ic_mean / ic_std if ic_std > 0 else np.nan
    t_stat = ic_mean / (ic_std / np.sqrt(len(ic_df))) if ic_std > 0 else np.nan

    print(f"[RankIC] 周度样本数: {len(ic_df)}")
    print(f"[RankIC] 均值: {ic_mean:.4f}, 标准差: {ic_std:.4f}, ICIR: {icir:.3f}, t值: {t_stat:.2f}")

    # 年度统计
    yearly = ic_df.copy()
    yearly["year"] = yearly.index.year
    grp = yearly.groupby("year")["rank_ic"]
    stats_df = pd.DataFrame({
        "n_weeks": grp.size(),
        "ic_mean": grp.mean(),
        "ic_std": grp.std(ddof=1)
    })
    stats_df["icir"] = stats_df["ic_mean"] / stats_df["ic_std"]
    # 让后续画图能拿到整体均值
    stats_df.attrs["overall_mean"] = ic_mean

    print("\n[RankIC][年度统计]")
    for y, r in stats_df.iterrows():
        print(f"  {int(y)}  n={int(r['n_weeks'])}  mean={r['ic_mean']:.4f}  std={r['ic_std']:.4f}  ICIR={r['icir']:.3f}")

    # 保存 CSV
    out_csv = out_dir / f"rankic_{cfg.run_name_out}.csv"
    ensure_dir(out_csv)
    ic_df.to_csv(out_csv, float_format="%.8f")
    print(f"[RankIC] 序列已保存：{out_csv}")

    out_year_csv = out_dir / f"yearly_rankic_stats_{cfg.run_name_out}.csv"
    ensure_dir(out_year_csv)
    stats_df.to_csv(out_year_csv, float_format="%.8f")
    print(f"[RankIC] 年度统计已保存：{out_year_csv}")

    # 画图：时间序列、直方图、年度均值柱状图
    ts_png = out_dir / f"rankic_timeseries_{cfg.run_name_out}.png"
    hist_png = out_dir / f"rankic_hist_{cfg.run_name_out}.png"
    yearly_png = out_dir / f"rankic_yearly_mean_{cfg.run_name_out}.png"
    plot_rankic_timeseries(ic_df, ts_png, title=f"Weekly RankIC ({cfg.run_name_out})")
    plot_rankic_hist(ic_df, hist_png, title=f"RankIC Distribution ({cfg.run_name_out})")
    plot_yearly_mean_bar(stats_df, yearly_png, title=f"Yearly RankIC Mean ({cfg.run_name_out})")
    print(f"[RankIC] 图片已保存：{ts_png}")
    print(f"[RankIC] 图片已保存：{hist_png}")
    print(f"[RankIC] 图片已保存：{yearly_png}")


if __name__ == "__main__":
    main()