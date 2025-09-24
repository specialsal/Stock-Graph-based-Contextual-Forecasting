# coding: utf-8
"""
optimize_position.py
统一“权重生成器”的入口：根据 BT_ROLL_CFG.weight_mode 生成持仓文件。
- weight_mode ∈ {"equal", "score", "optimize"}
- 输入：predictions_filtered_{run_name}.parquet（池内打分）
- 输出：positions_{run_name}_{weight_mode}.csv（仅含权重，不含 next_week_ret）
说明：
- equal/score：先选后权重（与原 btr_backtest 的等权/按分加权逻辑一致），不需要历史权重与协方差；
- optimize：L2 风险 + L2 调仓 QP（复用 optim_l2qp.py），需要上一周权重（若无则视为0）与对角协方差。
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd

from backtest_rolling_config import BT_ROLL_CFG as CFG_BT
from optim_l2qp import build_diag_cov, align_and_select, solve_portfolio_l2qp


def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def _load_scores(cfg) -> pd.DataFrame:
    # 读取池内打分
    p = cfg.backtest_dir / f"predictions_filtered_{cfg.run_name_out}.parquet"
    if not p.exists():
        alt = cfg.backtest_dir / f"predictions_filtered_{cfg.run_name}.parquet"
        p = alt if alt.exists() else p
    if not p.exists():
        raise FileNotFoundError(f"未找到池内打分文件：{p}")
    df = pd.read_parquet(p)
    df = df[(df["date"] >= pd.Timestamp(cfg.bt_start_date)) & (df["date"] <= pd.Timestamp(cfg.bt_end_date))]
    if df.empty:
        raise RuntimeError("回测区间内池内打分为空")
    # 规范类型
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["date", "stock", "score"])
    return df[["date", "stock", "score"]].copy()


def _select_top(df_d: pd.DataFrame, top_pct: float, max_n: int, min_n: int, filter_negative_scores_long: bool, weight_mode: str):
    # 先选后权重的通用选股与权重分配
    df_d = df_d.sort_values(["score", "stock"], ascending=[False, True])
    n = len(df_d)
    n_target = max(0, min(int(math.floor(n * top_pct)), int(max_n)))
    if n_target == 0 and n >= min_n:
        n_target = min(int(min_n), n)
    df_sel = df_d.head(n_target).copy()
    if filter_negative_scores_long:
        df_sel = df_sel[df_sel["score"] > 0]
    if len(df_sel) < min_n:
        return pd.DataFrame(columns=["stock", "weight", "score"])
    if weight_mode == "equal":
        w = np.full(len(df_sel), 1.0 / len(df_sel), dtype=float)
    else:
        s = df_sel["score"].values.astype(float)
        s = np.clip(s, 0.0, None)
        ssum = s.sum()
        w = (np.full(len(df_sel), 1.0 / len(df_sel), dtype=float) if ssum <= 0 else s / ssum)
    df_sel = df_sel.assign(weight=w)
    return df_sel[["stock", "weight", "score"]]


def _load_prev_weights(pos_path: Path, date: pd.Timestamp) -> pd.Series:
    if not pos_path.exists():
        return pd.Series(dtype=float, name="w_prev")
    df = pd.read_csv(pos_path)
    if df.empty or "date" not in df.columns or "stock" not in df.columns or "weight" not in df.columns:
        return pd.Series(dtype=float, name="w_prev")
    df["date"] = pd.to_datetime(df["date"])
    prev_dates = sorted([d for d in df["date"].unique() if pd.Timestamp(d) < date])
    if not prev_dates:
        return pd.Series(dtype=float, name="w_prev")
    d_prev = prev_dates[-1]
    dfp = df[df["date"] == d_prev]
    return pd.Series(dfp["weight"].values.astype(float), index=dfp["stock"].astype(str).values, name="w_prev")


def _load_returns_matrix(cfg, end_date: pd.Timestamp, lookback_days: int = 120) -> pd.DataFrame:
    price_df = pd.read_parquet(cfg.price_day_file)
    if not isinstance(price_df.index, pd.MultiIndex):
        price_df = price_df.set_index(["order_book_id", "date"]).sort_index()
    close = price_df["close"].unstack(0).sort_index()
    close = close[close.index <= end_date].tail(lookback_days + 1)
    rets = close.pct_change().dropna(how="all")
    return rets


def _scale_scores_to_returns(z: pd.Series, target_ic: float = 0.05) -> pd.Series:
    if z.empty or z.std(ddof=0) == 0:
        return z * 0.0
    z_std = (z - z.mean()) / (z.std(ddof=0) + 1e-12)
    return target_ic * z_std


def main():
    cfg = CFG_BT
    out_dir = cfg.backtest_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    weight_mode = str(getattr(cfg, "weight_mode", "equal")).lower()
    if weight_mode not in ("equal", "score", "optimize"):
        raise ValueError(f"weight_mode 不支持：{weight_mode}")

    # 输出文件：positions_{run_name}_{weight_mode}.csv
    pos_out = out_dir / f"positions_{cfg.run_name_out}_{weight_mode}.csv"
    ensure_dir(pos_out)

    # 读取池内打分
    scores_all = _load_scores(cfg)
    weeks = sorted(scores_all["date"].unique())

    results = []
    # 若 optimize 需要上一周权重，上一周文件路径为当前即将输出的统一文件（允许不存在）
    prev_pos_path = pos_out if pos_out.exists() else out_dir / f"positions_{cfg.run_name_out}_{weight_mode}.csv"

    for date in weeks:
        date = pd.Timestamp(date)
        sub = scores_all[scores_all["date"] == date].copy()
        if sub.empty:
            continue

        if weight_mode in ("equal", "score"):
            df_sel = _select_top(
                sub, top_pct=float(cfg.top_pct), max_n=int(cfg.max_n_stocks), min_n=int(cfg.min_n_stocks),
                filter_negative_scores_long=bool(cfg.filter_negative_scores_long),
                weight_mode=weight_mode
            )
            if df_sel.empty:
                # 记空仓
                pass
            else:
                for _, r in df_sel.iterrows():
                    results.append({"date": date, "stock": str(r["stock"]), "weight": float(r["weight"]), "score": float(r["score"])})
        else:
            # optimize 路径
            # 底池
            universe = sub["stock"].astype(str).tolist()
            # 上一周权重
            w_prev = _load_prev_weights(prev_pos_path, date)
            # 协方差（对角）
            rets = _load_returns_matrix(cfg, end_date=date, lookback_days=120).reindex(columns=universe)
            sigma_diag = build_diag_cov(rets).reindex(universe).fillna(rets.var().mean()).values
            # 打分 -> 预期收益尺度
            z_raw = pd.Series(sub["score"].values.astype(float), index=sub["stock"].astype(str).values, name="z")
            z = _scale_scores_to_returns(z_raw, target_ic=0.05)
            # 单票上限
            ub = pd.Series(0.05, index=universe)
            # 对齐与裁剪（提速）
            idx, z_vec, wprev_vec, ub_vec = align_and_select(universe=universe, z=z, w_prev=w_prev, ub=ub, top_n=300)
            # 对齐协方差
            pos_index = {s: k for k, s in enumerate(universe)}
            sigma_vec = np.asarray(sigma_diag)[[pos_index[s] for s in idx]]
            # 求解 QP
            w_opt_vec, diag = solve_portfolio_l2qp(
                z=z_vec, sigma_diag=sigma_vec, w_prev=wprev_vec, ub=ub_vec,
                lambda_risk=5.0, gamma_tc=10.0,
                full_invest=False, tau2_budget=None,
                solver="OSQP", verbose=False
            )
            # 回填到股票层（未入选为0）
            w_opt = pd.Series(0.0, index=universe, dtype=float)
            w_opt.loc[idx] = w_opt_vec
            # 写入结果
            for s, w in w_opt.items():
                results.append({"date": date, "stock": str(s), "weight": float(w), "score": float(z_raw.get(s, np.nan))})

    if not results:
        raise RuntimeError("未能生成任何持仓记录，请检查打分与配置。")

    out_df = pd.DataFrame(results)
    out_df = out_df.sort_values(["date", "weight"], ascending=[True, False])
    out_df["date"] = pd.to_datetime(out_df["date"])
    out_df.to_csv(pos_out, index=False, float_format="%.8f")
    print(f"[OPT-POS] saved: {pos_out}")


if __name__ == "__main__":
    main()