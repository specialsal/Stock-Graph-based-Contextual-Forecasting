# coding: utf-8
"""
optimize_position.py
统一“权重生成器”的入口：根据 BT_ROLL_CFG.weight_mode 生成持仓文件。
- weight_mode ∈ {"equal", "score", "optimize"}
- 输入：predictions_filtered_{run_name}.parquet（池内打分）
- 输出：positions_{run_name}_{weight_mode}.csv（包含权重与审计列 score_neutral）
说明：
- equal/score：先选后权重（与原 btr_backtest 的等权/按分加权逻辑一致），不需要历史权重与协方差；
- optimize：L2 风险 + L2 调仓 QP（复用 optim_l2qp.py），需要上一周权重（若无则视为0）与对角协方差。
- 若开启行业中性化（BT_ROLL_CFG.neutralize_enable=True），将对每周分数进行横截面 OLS 残差中性化，
  并在输出 positions 中加列 score_neutral 以便审计。

新增：
- 支持低频调仓：通过 backtest_rolling_config.BT_ROLL_CFG.rebalance_every_k（1/2/3）。
  仅在“调仓周”重算权重；非调仓周直接沿用上一周权重，不做QP、不变更成分。
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd

from backtest_rolling_config import BT_ROLL_CFG as CFG_BT
from optim_l2qp import build_diag_cov, align_and_select, solve_portfolio_l2qp

# 复用行业映射工具
from utils import load_industry_map


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


def _select_top(df_d: pd.DataFrame, top_pct: float, max_n: int, min_n: int,
                filter_negative_scores_long: bool, weight_mode: str, col_use: str,
                retain_scores: bool = True):
    # 先选后权重的通用选股与权重分配；col_use 为用于排序/权重的分数字段（可能是中性化后的）
    df_d = df_d.sort_values([col_use, "stock"], ascending=[False, True])
    n = len(df_d)
    n_target = max(0, min(int(math.floor(n * top_pct)), int(max_n)))
    if n_target == 0 and n >= min_n:
        n_target = min(int(min_n), n)
    df_sel = df_d.head(n_target).copy()

    if filter_negative_scores_long:
        df_sel = df_sel[df_sel[col_use] > 0]

    if len(df_sel) < min_n:
        # 返回空框架，包含需要的列
        base_cols = ["stock", "weight", "score"]
        if "score_neutral" in df_d.columns:
            base_cols.append("score_neutral")
        return pd.DataFrame(columns=base_cols)

    if weight_mode == "equal":
        w = np.full(len(df_sel), 1.0 / len(df_sel), dtype=float)
    else:
        s = df_sel[col_use].values.astype(float)
        s = np.clip(s, 0.0, None)  # 非负化
        ssum = s.sum()
        w = (np.full(len(df_sel), 1.0 / len(df_sel), dtype=float) if ssum <= 0 else s / ssum)

    df_sel = df_sel.assign(weight=w)

    # 返回时保留原 score 与 score_neutral（若有）
    cols_out = ["stock", "weight", "score"]
    if retain_scores and "score_neutral" in df_sel.columns:
        cols_out.append("score_neutral")
    return df_sel[cols_out]


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


def _winsorize_series(s: pd.Series, pct: float) -> pd.Series:
    # 对分数进行对称分位裁剪；pct=0 表示无裁剪
    pct = float(pct or 0.0)
    if pct <= 0.0:
        return s
    lo = s.quantile(pct)
    hi = s.quantile(1.0 - pct)
    return s.clip(lower=lo, upper=hi)


def _neutralize_week_scores(sub: pd.DataFrame, ind_map: dict, add_intercept: bool,
                            clip_pct: float, method: str = "ols_resid") -> pd.DataFrame:
    """
    对单周的打分做行业中性化：返回包含新列 score_neutral 的副本。
    - sub: 包含列 [date, stock, score]
    - ind_map: {stock -> industry_id}
    - method: "ols_resid"（默认）或 "group_demean"（降级兜底）
    """
    if sub is None or sub.empty:
        return sub.assign(score_neutral=np.nan)

    df = sub.copy()
    # 行业ID映射
    ind_id = df["stock"].map(lambda x: ind_map.get(str(x).strip(), None))
    # 缺失行业归为未知ID = max_id + 1
    max_id = max(ind_map.values()) if len(ind_map) > 0 else -1
    unk_id = max_id + 1
    ind_id = ind_id.fillna(unk_id).astype(int)
    df["ind_id"] = ind_id

    # 可选 winsorize
    sc = pd.to_numeric(df["score"], errors="coerce")
    sc = _winsorize_series(sc, pct=clip_pct)
    df["score_w"] = sc

    # 横截面 OLS 残差
    if method == "ols_resid":
        # one-hot 哑变量（避免稀疏行业影响；pandas.get_dummies 会自动去除缺失）
        X = pd.get_dummies(df["ind_id"].astype(int), prefix="ind", drop_first=False)
        if add_intercept:
            X = pd.concat([pd.Series(1.0, index=X.index, name="intercept"), X], axis=1)

        y = df["score_w"].astype(float)
        # 过滤非有限值样本
        mask = y.replace([np.inf, -np.inf], np.nan).notna()
        if mask.sum() < 2 or X.shape[1] == 0:
            # 兜底：分组去均值
            grp_mean = df.groupby("ind_id")["score_w"].transform("mean")
            df["score_neutral"] = (df["score_w"] - grp_mean).astype(float)
            return df.drop(columns=["ind_id", "score_w"])

        Xv = X.values.astype(float)
        yv = y.values.astype(float)
        # 最小二乘解
        try:
            beta, *_ = np.linalg.lstsq(Xv[mask.values], yv[mask.values], rcond=None)
            yhat = Xv @ beta
            resid = yv - yhat
            df["score_neutral"] = resid.astype(float)
        except Exception:
            # 回退：分组去均值
            grp_mean = df.groupby("ind_id")["score_w"].transform("mean")
            df["score_neutral"] = (df["score_w"] - grp_mean).astype(float)
    else:
        # 直接分组去均值
        grp_mean = df.groupby("ind_id")["score_w"].transform("mean")
        df["score_neutral"] = (df["score_w"] - grp_mean).astype(float)

    return df.drop(columns=["ind_id", "score_w"])


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

    # 若开启行业中性化，准备行业字典
    do_neutral = bool(getattr(cfg, "neutralize_enable", False))
    neutral_method = str(getattr(cfg, "neutralize_method", "ols_resid")).lower()
    add_intercept = bool(getattr(cfg, "neutralize_add_intercept", True))
    clip_pct = float(getattr(cfg, "neutralize_clip_pct", 0.0))

    ind_map = None
    if do_neutral:
        ind_map = load_industry_map(cfg.industry_map_file)  # {stock -> industry_id}

    # 调仓频率（1=每周；2=双周；3=三周）
    k = max(1, int(getattr(cfg, "rebalance_every_k", 1)))

    results = []
    # 若 optimize 需要上一周权重，上一周文件路径为当前即将输出的统一文件（允许不存在）
    prev_pos_path = pos_out if pos_out.exists() else out_dir / f"positions_{cfg.run_name_out}_{weight_mode}.csv"

    # 用于非调仓周直接沿用上一周权重（避免每次都从磁盘加载）
    last_week_weights = None  # pd.Series(index=stock, dtype=float)
    last_week_scores = None   # pd.Series(index=stock, dtype=float)
    last_week_scores_neu = None  # pd.Series(index=stock, dtype=float)

    for idx, date in enumerate(weeks):
        date = pd.Timestamp(date)
        sub = scores_all[scores_all["date"] == date].copy()
        if sub.empty:
            continue

        # 行业中性化（如开启）
        col_use = "score"
        if do_neutral:
            try:
                sub = _neutralize_week_scores(
                    sub=sub,
                    ind_map=ind_map,
                    add_intercept=add_intercept,
                    clip_pct=clip_pct,
                    method=neutral_method
                )
                if "score_neutral" in sub.columns and sub["score_neutral"].notna().any():
                    col_use = "score_neutral"
                else:
                    sub["score_neutral"] = sub["score"]
            except Exception:
                sub["score_neutral"] = sub["score"]
                col_use = "score"
        else:
            sub["score_neutral"] = sub["score"]

        is_rebalance = ((idx % k) == 0)

        if not is_rebalance:
            # 非调仓周：沿用上一周权重。如果上一周权重为空，回退为调仓
            if last_week_weights is None or last_week_weights.empty:
                is_rebalance = True

        if is_rebalance:
            # 计算当周权重
            if weight_mode in ("equal", "score"):
                df_sel = _select_top(
                    sub.assign(score=sub["score"].astype(float)),
                    top_pct=float(cfg.top_pct), max_n=int(cfg.max_n_stocks), min_n=int(cfg.min_n_stocks),
                    filter_negative_scores_long=bool(cfg.filter_negative_scores_long),
                    weight_mode=weight_mode,
                    col_use=col_use,
                    retain_scores=True
                )
                if df_sel.empty:
                    # 本周空仓：不写入记录
                    last_week_weights = None
                    last_week_scores = None
                    last_week_scores_neu = None
                    continue
                # 写入并缓存本周结果
                w_ser = pd.Series(df_sel["weight"].values.astype(float), index=df_sel["stock"].astype(str).values)
                last_week_weights = w_ser.copy()
                last_week_scores = pd.Series(df_sel["score"].values.astype(float), index=df_sel["stock"].astype(str).values)
                if "score_neutral" in df_sel.columns:
                    last_week_scores_neu = pd.Series(df_sel["score_neutral"].values.astype(float), index=df_sel["stock"].astype(str).values)
                else:
                    last_week_scores_neu = last_week_scores.copy()

                for s, w in w_ser.items():
                    results.append({
                        "date": date,
                        "stock": str(s),
                        "weight": float(w),
                        "score": float(last_week_scores.get(s, np.nan)),
                        "score_neutral": float(last_week_scores_neu.get(s, np.nan))
                    })

            else:
                # optimize 路径（调仓周才做QP）
                universe = sub["stock"].astype(str).tolist()
                w_prev = _load_prev_weights(prev_pos_path, date)
                rets = _load_returns_matrix(cfg, end_date=date, lookback_days=120).reindex(columns=universe)
                sigma_diag = build_diag_cov(rets).reindex(universe).fillna(rets.var().mean()).values

                z_raw = pd.Series(
                    (sub[col_use] if col_use in sub.columns else sub["score"]).values.astype(float),
                    index=sub["stock"].astype(str).values,
                    name="z"
                )
                z_audit_neu = pd.Series(sub["score_neutral"].values.astype(float), index=sub["stock"].astype(str).values)
                z = _scale_scores_to_returns(z_raw, target_ic=0.05)

                ub = pd.Series(0.05, index=universe)
                idx_keep, z_vec, wprev_vec, ub_vec = align_and_select(universe=universe, z=z, w_prev=w_prev, ub=ub, top_n=300)
                pos_index = {s: k for k, s in enumerate(universe)}
                sigma_vec = np.asarray(sigma_diag)[[pos_index[s] for s in idx_keep]]

                w_opt_vec, diag = solve_portfolio_l2qp(
                    z=z_vec, sigma_diag=sigma_vec, w_prev=wprev_vec, ub=ub_vec,
                    lambda_risk=5.0, gamma_tc=10.0,
                    full_invest=False, tau2_budget=None,
                    solver="OSQP", verbose=False
                )
                w_opt = pd.Series(0.0, index=universe, dtype=float)
                w_opt.loc[idx_keep] = w_opt_vec

                # 缓存当周权重与审计列
                last_week_weights = w_opt.copy()
                last_week_scores = z_raw.copy()
                last_week_scores_neu = z_audit_neu.copy()

                for s, w in w_opt.items():
                    results.append({
                        "date": date,
                        "stock": str(s),
                        "weight": float(w),
                        "score": float(z_raw.get(s, np.nan)),
                        "score_neutral": float(z_audit_neu.get(s, np.nan))
                    })

        else:
            # 非调仓周：沿用上一周权重与审计分数（不做任何重新计算/选股/QP）
            # last_week_weights 至少非空
            for s, w in last_week_weights.items():
                results.append({
                    "date": date,
                    "stock": str(s),
                    "weight": float(w),
                    "score": float(last_week_scores.get(s, np.nan)) if last_week_scores is not None else np.nan,
                    "score_neutral": float(last_week_scores_neu.get(s, np.nan)) if last_week_scores_neu is not None else (
                        float(last_week_scores.get(s, np.nan)) if last_week_scores is not None else np.nan
                    )
                })

    if not results:
        raise RuntimeError("未能生成任何持仓记录，请检查打分与配置。")

    out_df = pd.DataFrame(results)
    out_df = out_df.sort_values(["date", "weight"], ascending=[True, False])
    out_df["date"] = pd.to_datetime(out_df["date"])
    # 确保包含审计列
    if "score_neutral" not in out_df.columns:
        out_df["score_neutral"] = out_df["score"]

    out_df.to_csv(pos_out, index=False, float_format="%.8f")
    print(f"[OPT-POS] saved: {pos_out}")


if __name__ == "__main__":
    main()