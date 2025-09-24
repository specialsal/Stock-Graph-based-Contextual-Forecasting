# coding: utf-8
"""
compute_turnover.py
计算周度组合换手率（基于权重）、可视化并输出年度统计。

定义
- 输入：backtest_rolling/{run_name}/positions_{run_name}.csv
  需要列：date, stock, weight
- 周度换手率（%）：turnover_t = 0.5 * sum_i |w_t(i) - w_{t-1}(i)|
  注：当 t 与 t-1 完全无交集且都满仓（权重和为1），则 turnover=100%。

输出
- backtest_rolling/{run_name}/turnover_{run_name}.csv        （date, turnover）
- backtest_rolling/{run_name}/turnover_timeseries_{run_name}.png
- backtest_rolling/{run_name}/turnover_hist_{run_name}.png
- backtest_rolling/{run_name}/yearly_turnover_stats_{run_name}.csv
- backtest_rolling/{run_name}/turnover_yearly_mean_{run_name}.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from backtest_rolling_config import BT_ROLL_CFG


def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def load_positions(cfg) -> pd.DataFrame:
    out_dir = cfg.backtest_dir
    pos_path = out_dir / f"positions_{cfg.run_name_out}_{cfg.weight_mode}.csv"
    if not pos_path.exists():
        alt = out_dir / f"positions_{cfg.run_name}.csv"
        if alt.exists():
            pos_path = alt
    if not pos_path.exists():
        raise FileNotFoundError(f"未找到持仓文件：{pos_path}")

    df = pd.read_csv(pos_path)
    # 规范列名
    need_cols = {"date", "stock", "weight"}
    miss = need_cols - set(df.columns)
    if miss:
        raise RuntimeError(f"positions 缺少必要列：{miss}")
    # 解析日期
    df["date"] = pd.to_datetime(df["date"])
    # 仅保留回测区间
    df = df[(df["date"] >= pd.Timestamp(cfg.bt_start_date)) & (df["date"] <= pd.Timestamp(cfg.bt_end_date))]
    # 清理权重异常
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df = df.dropna(subset=["weight"])
    return df


def pivot_weights(positions: pd.DataFrame) -> pd.DataFrame:
    """
    将 (date, stock, weight) 明细转为以 date 为行、stock 为列的权重矩阵，缺失填 0。
    同时对每个日期做权重归一化，避免上游轻微数值误差。
    """
    W = positions.pivot_table(index="date", columns="stock", values="weight", aggfunc="sum").sort_index()
    W = W.fillna(0.0)
    # 对每行归一化（若合计为0则保持0）
    s = W.sum(axis=1)
    nonzero = s > 0
    W.loc[nonzero] = (W.loc[nonzero].T / s[nonzero]).T
    return W


def compute_weekly_turnover(W: pd.DataFrame) -> pd.Series:
    """
    输入：W — 权重矩阵（index=date, columns=stock, 值为权重，行内已归一化）
    输出：周度换手率（0~1），index 对齐 date，从第二个日期开始有值。
    turnover_t = 0.5 * sum |w_t - w_{t-1}|
    """
    dates = W.index.to_list()
    turns = []
    for i in range(1, len(dates)):
        w_prev = W.iloc[i - 1].values
        w_curr = W.iloc[i].values
        # 对齐列集合（已对齐，因为来自同一透视矩阵）
        diff = np.abs(w_curr - w_prev).sum()
        turn = 0.5 * diff
        # 在数值误差范围内裁剪
        turn = float(np.clip(turn, 0.0, 1.0))
        turns.append({"date": dates[i], "turnover": turn})
    ts = pd.DataFrame(turns).set_index("date")["turnover"]
    return ts


def plot_turnover_timeseries(ts: pd.Series, out_path: Path, title: str = "Weekly Turnover"):
    plt.figure(figsize=(10, 4.5), dpi=160)
    x = ts.index
    y = ts.values
    mean_val = np.nanmean(y)

    plt.plot(x, y, color="#2F6CFF", linewidth=1.4, label="Turnover")
    plt.axhline(mean_val, color="#FF7F0E", linewidth=1.2, linestyle="--", label=f"Mean = {mean_val:.2%}")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Turnover")
    plt.ylim(0, min(1.0, max(0.6, np.nanmax(y) * 1.1)) if np.isfinite(np.nanmax(y)) else 1.0)
    plt.legend(loc="best", frameon=False)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    ensure_dir(out_path)
    plt.savefig(out_path)
    plt.close()


def plot_turnover_hist(ts: pd.Series, out_path: Path, title: str = "Turnover Distribution"):
    plt.figure(figsize=(6.5, 4.5), dpi=160)
    y = ts.dropna().values
    mean_val = np.nanmean(y)

    plt.hist(y, bins=30, color="#2F6CFF", alpha=0.75, edgecolor="white")
    plt.axvline(mean_val, color="#FF7F0E", linewidth=1.2, linestyle="--", label=f"Mean = {mean_val:.2%}")
    plt.title(title)
    plt.xlabel("Turnover")
    plt.ylabel("Frequency")
    plt.legend(loc="best", frameon=False)
    plt.grid(True, linestyle="--", alpha=0.25)
    plt.tight_layout()
    ensure_dir(out_path)
    plt.savefig(out_path)
    plt.close()


def plot_yearly_mean_bar(stats_df: pd.DataFrame, out_path: Path, title: str = "Yearly Average Turnover"):
    plt.figure(figsize=(8.5, 4.5), dpi=160)
    years = stats_df.index.astype(str).tolist()
    means = stats_df["turnover_mean"].values
    overall_mean = stats_df.attrs.get("overall_mean", np.nan)

    bars = plt.bar(years, means, color="#2F6CFF", alpha=0.85, edgecolor="white")
    if np.isfinite(overall_mean):
        plt.axhline(overall_mean, color="#FF7F0E", linewidth=1.2, linestyle="--", label=f"Overall Mean = {overall_mean:.2%}")
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("Average Turnover")
    for b, v in zip(bars, means):
        plt.text(b.get_x() + b.get_width()/2, b.get_height() + (0.003 if v>=0 else -0.003),
                 f"{v:.2%}", ha="center", va="bottom" if v>=0 else "top", fontsize=9, color="#333333")
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

    # 读取持仓
    positions = load_positions(cfg)
    if positions.empty:
        raise RuntimeError("positions 明细为空，无法计算换手率")

    # 仅保留必要列
    positions = positions[["date", "stock", "weight"]]
    # 排序
    positions = positions.sort_values(["date", "stock"])

    # 透视为权重矩阵并归一化
    W = pivot_weights(positions)
    if W.shape[0] < 2:
        raise RuntimeError("有效周度样本不足（少于2周），无法计算换手率")

    # 计算周度换手率
    ts = compute_weekly_turnover(W)

    # 整体统计
    mean_turn = ts.mean()
    std_turn = ts.std(ddof=1)
    print(f"[Turnover] 周度样本数: {len(ts)}")
    print(f"[Turnover] 均值: {mean_turn:.2%}, 标准差: {std_turn:.2%}")

    # 年度统计
    df_ts = ts.to_frame(name="turnover")
    df_ts["year"] = df_ts.index.year
    grp = df_ts.groupby("year")["turnover"]
    stats_df = pd.DataFrame({
        "n_weeks": grp.size(),
        "turnover_mean": grp.mean(),
        "turnover_std": grp.std(ddof=1)
    })
    stats_df.attrs["overall_mean"] = mean_turn

    print("\n[Turnover][年度统计]")
    for y, r in stats_df.iterrows():
        print(f"  {int(y)}  n={int(r['n_weeks'])}  mean={r['turnover_mean']:.2%}  std={r['turnover_std']:.2%}")

    # 保存 CSV
    out_csv = out_dir / f"turnover_{cfg.run_name_out}.csv"
    ensure_dir(out_csv)
    df_ts[["turnover"]].to_csv(out_csv, float_format="%.6f")
    print(f"[Turnover] 序列已保存：{out_csv}")

    out_year_csv = out_dir / f"yearly_turnover_stats_{cfg.run_name_out}.csv"
    ensure_dir(out_year_csv)
    stats_df.to_csv(out_year_csv, float_format="%.6f")
    print(f"[Turnover] 年度统计已保存：{out_year_csv}")

    # 画图
    ts_png = out_dir / f"turnover_timeseries_{cfg.run_name_out}.png"
    hist_png = out_dir / f"turnover_hist_{cfg.run_name_out}.png"
    yearly_png = out_dir / f"turnover_yearly_mean_{cfg.run_name_out}.png"
    plot_turnover_timeseries(ts, ts_png, title=f"Weekly Turnover ({cfg.run_name_out})")
    plot_turnover_hist(ts, hist_png, title=f"Turnover Distribution ({cfg.run_name_out})")
    plot_yearly_mean_bar(stats_df, yearly_png, title=f"Yearly Average Turnover ({cfg.run_name_out})")
    print(f"[Turnover] 图片已保存：{ts_png}")
    print(f"[Turnover] 图片已保存：{hist_png}")
    print(f"[Turnover] 图片已保存：{yearly_png}")


if __name__ == "__main__":
    main()