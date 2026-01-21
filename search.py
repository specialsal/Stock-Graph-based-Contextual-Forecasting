# coding: utf-8
"""
网格搜索止盈/止损 + 冷静期参数，仅关注 overall["sharpe"]，
并在 (sl_price_ratio × cooldown_days) 平面上绘制 Sharpe 热力图。

运行方式：
    python grid_search_stops_sharpe.py
"""

import json
from pathlib import Path
from copy import deepcopy
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from backtest_rolling_config import BT_ROLL_CFG
import btr_backtest
import btr_metrics


def run_once_and_get_sharpe() -> float:
    """
    跑一次回测 + 指标，返回 metrics_xxx.json 中 overall["sharpe"]。
    若读取失败或不存在则返回 np.nan。
    """
    cfg = BT_ROLL_CFG
    out_dir = cfg.backtest_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 跑回测
    print(f"[GRID] Running backtest with tp={cfg.tp_price_ratio}, "
          f"sl={cfg.sl_price_ratio}, cd={cfg.cooldown_days}")
    btr_backtest.main()

    # 2) 指标计算
    nav_path = out_dir / f"nav_{cfg.run_name_out}.csv"
    if not nav_path.exists():
        raise FileNotFoundError(f"未找到 NAV 文件：{nav_path}")
    btr_metrics.main(str(nav_path))

    # 3) 读取 metrics_xxx.json
    run_name = cfg.run_name_out
    metrics_dir = nav_path.parent / f"metrics_{run_name}"
    metrics_json = metrics_dir / f"metrics_{run_name}.json"
    if not metrics_json.exists():
        raise FileNotFoundError(f"未找到指标文件：{metrics_json}")

    with open(metrics_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    overall = data.get("overall", {})
    sharpe = overall.get("sharpe", np.nan)
    try:
        sharpe = float(sharpe)
    except Exception:
        sharpe = np.nan
    return sharpe


def main():
    cfg = BT_ROLL_CFG

    # ===== 1. 备份原始配置 =====
    cfg_backup = deepcopy(cfg)

    # 确保开启周内止盈止损
    cfg.enable_intraweek_stops = True

    # ===== 2. 定义搜索网格 =====
    # 可以先用较小网格测试，确认流程 OK 后再放大
    tp_grid = [0.05,0.10, 0.15,0.2,99]          # 止盈
    sl_grid = [0.03, 0.05, 0.10,99]    # 止损
    cd_grid = [1, 3, 5, 10]                # 冷静期天数（自然日）

    results: List[Dict] = []

    # ===== 3. 网格循环 =====
    for tp in tp_grid:
        for sl in sl_grid:
            for cd in cd_grid:
                cfg.tp_price_ratio = float(tp)
                cfg.sl_price_ratio = float(sl)
                cfg.cooldown_days  = float(cd)

                try:
                    sharpe = run_once_and_get_sharpe()
                except Exception as e:
                    print(f"[GRID] 运行失败：tp={tp}, sl={sl}, cd={cd}, err={e}")
                    sharpe = np.nan

                results.append({
                    "tp_price_ratio": tp,
                    "sl_price_ratio": sl,
                    "cooldown_days": cd,
                    "sharpe": sharpe
                })

    # ===== 4. 恢复原始配置 =====
    for attr, val in cfg_backup.__dict__.items():
        setattr(cfg, attr, val)

    # ===== 5. 存结果表 =====
    res_df = pd.DataFrame(results)
    out_dir = cfg.backtest_dir / "grid_search_stops"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "grid_results_sharpe.csv"
    res_df.to_csv(csv_path, index=False, float_format="%.8f")
    print(f"[GRID] 网格搜索结果已保存到 {csv_path}")

    # ===== 6. 画 Sharpe 热力图 =====
    sns.set(style="whitegrid", font="SimHei", rc={"axes.unicode_minus": False})

    metric = "sharpe"

    for tp in tp_grid:
        sub = res_df[res_df["tp_price_ratio"] == tp].copy()
        if sub.empty:
            continue

        # 透视：行=sl_price_ratio，列=cooldown_days，值=sharpe
        pivot = sub.pivot(index="sl_price_ratio",
                          columns="cooldown_days",
                          values=metric)

        plt.figure(figsize=(6, 4))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",          # Sharpe 越大越好，用绿-红渐变
            cbar_kws={"shrink": 0.8}
        )
        plt.title(f"Sharpe 热力图 (tp_price_ratio={tp})")
        plt.xlabel("cooldown_days")
        plt.ylabel("sl_price_ratio")
        plt.tight_layout()

        png_path = out_dir / f"heatmap_sharpe_tp{tp}.png"
        plt.savefig(png_path, dpi=150)
        plt.close()
        print(f"[GRID] 已保存热力图：{png_path}")


if __name__ == "__main__":
    main()