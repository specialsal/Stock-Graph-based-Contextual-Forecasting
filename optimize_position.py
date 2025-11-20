# coding: utf-8
"""
optimize_position.py
生成“日度持仓”（支持低频调仓 rebalance_every_k）：
- weight_mode ∈ {"equal", "score"}
- 输入：predictions_filtered_{run_name}.parquet（或 run_name_out 版本），包含池内在各“信号周五”的打分
- 步骤：
  1) 逐“信号周五”处理：
     - 若为“调仓周”（按 rebalance_every_k），用该周打分选股 + 赋权；
     - 否则沿用“最近一次调仓周”的持仓与权重。
  2) 用交易日历将相邻两次“信号周”之间的所有交易日，逐日复制“最近一次调仓周”的权重。
  3) 输出为日度持仓 CSV：positions_{run_name_out}_{weight_mode}.csv，包含列：
     [date, stock, weight, score, score_neutral, score_neutral_mkt, signal_date]

中性化逻辑：
- 行业中性化：
    - neutralize_enable=True 时，对单周 score 做行业 OLS 残差，得到 score_neutral。
- 市值中性化：
    - mkt_neutral_enable=True 时，对“基准分数列”按 log(市值) 分箱，然后箱内去均值：
        * 若行业中性化打开：基准列为 score_neutral，产物列记为 score_neutral_mkt
        * 若行业中性化关闭：基准列为 score，产物列记为 score_mkt_neutral（并同时赋值给 score_neutral_mkt）

最终排序 / 赋权使用的列 col_use：
- 行业关 + 市值关：col_use = "score"
- 行业开 + 市值关：col_use = "score_neutral"
- 行业关 + 市值开：col_use = "score_neutral_mkt"
- 行业开 + 市值开：col_use = "score_neutral_mkt"
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd

from backtest_rolling_config import BT_ROLL_CFG as CFG_BT
from utils import load_industry_map, load_calendar
from config import CFG


def ensure_dir(p: Path):
    """确保输出目录存在"""
    p.parent.mkdir(parents=True, exist_ok=True)


def _load_scores(cfg) -> pd.DataFrame:
    """
    读取池内打分（信号周五）：
    - 优先读取 predictions_filtered_{run_name_out}.parquet
    - 若不存在，则回退到 predictions_filtered_{run_name}.parquet
    """
    p = cfg.backtest_dir / f"predictions_filtered_{cfg.run_name_out}.parquet"
    if not p.exists():
        alt = cfg.backtest_dir / f"predictions_filtered_{cfg.run_name}.parquet"
        p = alt if alt.exists() else p
    df = pd.read_parquet(p)
    # 时间裁剪到回测区间
    df = df[(df["date"] >= pd.Timestamp(cfg.bt_start_date)) & (df["date"] <= pd.Timestamp(cfg.bt_end_date))]
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["date", "stock", "score"])
    return df[["date", "stock", "score"]].copy()


def _winsorize_series(s: pd.Series, pct: float) -> pd.Series:
    """对序列 s 按百分位 pct / (1-pct) 进行 winsorize 裁剪"""
    if pct <= 0.0:
        return s
    lo = s.quantile(pct)
    hi = s.quantile(1.0 - pct)
    return s.clip(lower=lo, upper=hi)


def _neutralize_week_scores(
    sub: pd.DataFrame,
    ind_map: dict,
    add_intercept: bool,
    clip_pct: float,
    method: str = "ols_resid",
) -> pd.DataFrame:
    """
    对单周（信号周五）的打分做“行业中性化”，输出增加 score_neutral 列。

    实现：
    - 构造行业 ID（ind_id）；
    - 对 score 做 winsorize（clip_pct）；
    - 若 method="ols_resid"：对 score_w ~ 行业哑变量 回归，取残差作为 score_neutral；
    - 否则：对每个行业内做 group_demean（score_w - 行业内均值）。
    """
    df = sub.copy()
    # 行业ID映射
    max_id = max(ind_map.values()) if len(ind_map) > 0 else -1
    unk_id = max_id + 1
    ind_id = df["stock"].map(lambda x: ind_map.get(str(x).strip(), unk_id)).astype(int)
    df["ind_id"] = ind_id

    sc = pd.to_numeric(df["score"], errors="coerce")
    sc = _winsorize_series(sc, pct=clip_pct)
    df["score_w"] = sc

    if method == "ols_resid":
        # 行业 one-hot
        X = pd.get_dummies(df["ind_id"].astype(int), prefix="ind", drop_first=False)
        if add_intercept:
            X = pd.concat([pd.Series(1.0, index=X.index, name="intercept"), X], axis=1)
        y = df["score_w"].astype(float).values
        Xv = X.values.astype(float)
        # 最小二乘
        beta, *_ = np.linalg.lstsq(Xv, y, rcond=None)
        yhat = Xv @ beta
        resid = y - yhat
        df["score_neutral"] = resid.astype(float)
    else:
        # group_demean：按行业分组减去组内均值
        grp_mean = df.groupby("ind_id")["score_w"].transform("mean")
        df["score_neutral"] = (df["score_w"] - grp_mean).astype(float)

    return df.drop(columns=["ind_id", "score_w"])


def _neutralize_week_scores_mkt(
    sub: pd.DataFrame,
    mkt_df: pd.DataFrame,
    sig_date: pd.Timestamp,
    base_col: str,
    n_bins: int,
    clip_pct: float,
) -> pd.DataFrame:
    """
    对单周（信号周五）的打分做“市值中性化”（分箱去均值），返回增加市值中性列的 DataFrame。

    参数：
    - sub: 本周打分 DataFrame，需至少包含 [stock, score, score_neutral(可选)]
    - mkt_df: 市值 DataFrame，MultiIndex (order_book_id, date)，包含列 market_cap_2
    - sig_date: 信号日期（一般为周五交易日）
    - base_col: 作为被中性化的“基准分数列”：
        * 若行业中性化开启：base_col = "score_neutral"
        * 若行业中性化关闭：base_col = "score"
    - n_bins: log(市值) 分箱数量（qcut）
    - clip_pct: 对 log(市值) 做 winsorize 的百分位（0~0.5）

    输出：
    - 在 sub 基础上增加:
        * 若行业中性化开启 -> "score_neutral_mkt"
        * 若行业中性化关闭 -> "score_mkt_neutral"（并同时复制到 "score_neutral_mkt"，统一下游接口）
    """
    df = sub.copy()

    # 若基准列不存在，直接返回
    if base_col not in df.columns:
        return df

    # 从市值表中提取当日 sig_date 的市值切片
    try:
        mkt_sub = mkt_df.xs(sig_date, level="date")
    except Exception:
        return df

    if "market_cap_2" not in mkt_sub.columns:
        return df

    # 代码统一成字符串
    df["stock"] = df["stock"].astype(str).str.strip()
    mkt_sub = mkt_sub.reset_index()
    mkt_sub["order_book_id"] = mkt_sub["order_book_id"].astype(str).str.strip()

    # 合并市值
    df = df.merge(
        mkt_sub[["order_book_id", "market_cap_2"]],
        left_on="stock",
        right_on="order_book_id",
        how="left",
    )
    df = df.drop(columns=["order_book_id"])

    if not df["market_cap_2"].notna().any():
        return df

    # log(市值) + winsorize
    mkt_raw = pd.to_numeric(df["market_cap_2"], errors="coerce")
    mkt_raw = mkt_raw.where(mkt_raw > 0)
    log_mkt = np.log(mkt_raw)

    if clip_pct > 0.0:
        log_mkt = _winsorize_series(log_mkt, pct=clip_pct)

    df["log_mkt"] = log_mkt

    valid_mask = df["log_mkt"].notna()
    if valid_mask.sum() < max(4, n_bins):
        return df

    # 分位数分箱
    try:
        df.loc[valid_mask, "mkt_bin"] = pd.qcut(
            df.loc[valid_mask, "log_mkt"],
            q=n_bins,
            labels=False,
            duplicates="drop",
        )
    except Exception:
        return df

    # 基准分数
    base = pd.to_numeric(df[base_col], errors="coerce")

    # 先整体创建列，明确 dtype 为 float64，避免后面赋值 dtype 不匹配
    df["score_neutral_mkt"] = base.astype(float)

    # 对每个市值箱内做 group_demean：score - group_mean
    if df["mkt_bin"].notna().any():
        # 这里也转成 float，保证 dtype 统一
        grp_mean = (
            df.loc[valid_mask]
            .groupby("mkt_bin")[base_col]
            .transform("mean")
            .astype(float)
        )
        idx_valid_bins = valid_mask & df["mkt_bin"].notna()

        # 显式构造要写入的 float64 数组，避免 FutureWarning
        adj_vals = (
            df.loc[idx_valid_bins, base_col].astype(float).values
            - grp_mean.loc[idx_valid_bins].values
        ).astype(float)

        df.loc[idx_valid_bins, "score_neutral_mkt"] = adj_vals

    # 若行业中性化关闭，则此列本身就是“纯市值中性分数”
    if base_col == "score":
        df["score_mkt_neutral"] = df["score_neutral_mkt"].astype(float)

    # 清理临时列
    df = df.drop(columns=["log_mkt", "mkt_bin", "market_cap_2"], errors="ignore")

    return df


def _select_and_weight(
    df_week: pd.DataFrame,
    weight_mode: str,
    top_pct: float,
    max_n: int,
    min_n: int,
    filter_negative_scores_long: bool,
    col_use: str,
) -> pd.DataFrame:
    """
    对单周信号做选股与权重分配。返回列：stock, weight, score, score_neutral, score_neutral_mkt
    """
    if col_use not in df_week.columns:
        return pd.DataFrame(
            columns=["stock", "weight", "score", "score_neutral", "score_neutral_mkt"]
        )

    d = df_week.sort_values([col_use, "stock"], ascending=[False, True]).copy()
    n = len(d)
    if n == 0:
        return pd.DataFrame(
            columns=["stock", "weight", "score", "score_neutral", "score_neutral_mkt"]
        )

    n_target = max(0, min(int(math.floor(n * top_pct)), int(max_n)))
    if n_target == 0 and n >= min_n:
        n_target = min(int(min_n), n)
    d = d.head(n_target)

    if filter_negative_scores_long:
        d = d[d[col_use] > 0]

    if len(d) < min_n:
        return pd.DataFrame(
            columns=["stock", "weight", "score", "score_neutral", "score_neutral_mkt"]
        )

    # 权重
    if weight_mode == "equal":
        w = np.full(len(d), 1.0 / len(d), dtype=float)
    else:
        s = d[col_use].values.astype(float)
        s = np.clip(s, 0.0, None)
        ssum = s.sum()
        if ssum <= 0:
            w = np.full(len(d), 1.0 / len(d), dtype=float)
        else:
            w = (s / ssum).astype(float)
    d = d.assign(weight=w.astype(float))

    # 确保必要列存在
    if "score_neutral" not in d.columns:
        d["score_neutral"] = d["score"].astype(float)
    if "score_neutral_mkt" not in d.columns:
        d["score_neutral_mkt"] = d["score_neutral"].astype(float)

    return d[["stock", "weight", "score", "score_neutral", "score_neutral_mkt"]]


def main():
    cfg = CFG_BT
    out_dir = cfg.backtest_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    weight_mode = str(getattr(cfg, "weight_mode", "score")).lower()
    if weight_mode not in ("equal", "score"):
        raise ValueError(f"weight_mode 仅支持 equal/score，当前={weight_mode}")

    reb_k = int(getattr(cfg, "rebalance_every_k", 1))
    if reb_k <= 0:
        reb_k = 1

    # 读取池内打分（信号周五）
    scores_all = _load_scores(cfg)
    weeks = sorted(pd.to_datetime(scores_all["date"].unique()))

    # ===== 行业中性化配置 =====
    do_neutral = bool(getattr(cfg, "neutralize_enable", False))
    neutral_method = str(getattr(cfg, "neutralize_method", "ols_resid")).lower()
    add_intercept = bool(getattr(cfg, "neutralize_add_intercept", True))
    clip_pct_ind = float(getattr(cfg, "neutralize_clip_pct", 0.0))
    ind_map = load_industry_map(cfg.industry_map_file) if do_neutral else {}

    # ===== 市值中性化配置 =====
    do_mkt_neutral = bool(getattr(cfg, "mkt_neutral_enable", False))
    mkt_n_bins = int(getattr(cfg, "mkt_neutral_n_bins", 5))
    mkt_clip_pct = float(getattr(cfg, "mkt_neutral_clip_pct", 0.0))

    # 若需要市值中性化，加载市值文件
    mkt_df = None
    if do_mkt_neutral:
        mkt_path = cfg.price_fundamental_file
        if not Path(mkt_path).exists():
            raise FileNotFoundError(f"启用了市值中性化，但未找到市值文件：{mkt_path}")
        mkt_df = pd.read_parquet(mkt_path)
        if not isinstance(mkt_df.index, pd.MultiIndex):
            # 兼容：若索引不是 MultiIndex，则尝试从列中设置
            if {"order_book_id", "date"}.issubset(mkt_df.columns):
                mkt_df["date"] = pd.to_datetime(mkt_df["date"])
                mkt_df = mkt_df.set_index(["order_book_id", "date"]).sort_index()
            else:
                raise ValueError(
                    "price_fundamental_file 需要为 MultiIndex(order_book_id, date) 或至少包含这些列。"
                )
        if "market_cap_2" not in mkt_df.columns:
            raise ValueError(
                "price_fundamental_file 中缺少列 'market_cap_2'，无法进行市值中性化。"
            )

    # 周度组合（含低频调仓逻辑）
    weekly_port = {}  # date -> DataFrame(stock, weight, score, score_neutral, score_neutral_mkt)
    last_reb_sig_date = None
    last_reb_port = None

    for i, d in enumerate(weeks):
        sub = scores_all[scores_all["date"] == d].copy()

        # ====== 1) 行业中性化（可选） ======
        if do_neutral:
            sub_n = _neutralize_week_scores(
                sub,
                ind_map,
                add_intercept=add_intercept,
                clip_pct=clip_pct_ind,
                method=neutral_method,
            )
        else:
            sub_n = sub.copy()
            sub_n["score_neutral"] = pd.to_numeric(sub_n["score"], errors="coerce")

        # ====== 2) 市值中性化（可选） ======
        base_col_for_mkt = "score_neutral" if do_neutral else "score"
        if do_mkt_neutral and mkt_df is not None:
            sub_n = _neutralize_week_scores_mkt(
                sub=sub_n,
                mkt_df=mkt_df,
                sig_date=pd.Timestamp(d),
                base_col=base_col_for_mkt,
                n_bins=mkt_n_bins,
                clip_pct=mkt_clip_pct,
            )
        else:
            sub_n["score_neutral_mkt"] = sub_n.get("score_neutral", sub_n["score"]).astype(float)

        # ====== 3) 本周是否触发调仓 ======
        trigger_reb = (i % reb_k == 0)

        if trigger_reb:
            # col_use 选择逻辑
            if do_mkt_neutral:
                col_use = "score_neutral_mkt"
            else:
                col_use = "score_neutral" if do_neutral else "score"

            dsel = _select_and_weight(
                df_week=sub_n.assign(
                    score=pd.to_numeric(sub_n["score"], errors="coerce")
                ),
                weight_mode=weight_mode,
                top_pct=float(cfg.top_pct),
                max_n=int(cfg.max_n_stocks),
                min_n=int(cfg.min_n_stocks),
                filter_negative_scores_long=bool(cfg.filter_negative_scores_long),
                col_use=col_use,
            )
            if not dsel.empty:
                weekly_port[pd.Timestamp(d)] = dsel
                last_reb_sig_date = pd.Timestamp(d)
                last_reb_port = dsel
            else:
                if last_reb_port is not None:
                    weekly_port[pd.Timestamp(d)] = last_reb_port.copy()
                else:
                    weekly_port[pd.Timestamp(d)] = pd.DataFrame(
                        columns=[
                            "stock",
                            "weight",
                            "score",
                            "score_neutral",
                            "score_neutral_mkt",
                        ]
                    )
        else:
            if last_reb_port is not None:
                weekly_port[pd.Timestamp(d)] = last_reb_port.copy()
            else:
                weekly_port[pd.Timestamp(d)] = pd.DataFrame(
                    columns=[
                        "stock",
                        "weight",
                        "score",
                        "score_neutral",
                        "score_neutral_mkt",
                    ]
                )

    # ===== 将周度组合展开为“日度持仓” =====
    if hasattr(cfg, "trading_day_file"):
        try:
            cal_df = pd.read_csv(cfg.trading_day_file, header=None, skiprows=1)
            trading_days = pd.DatetimeIndex(pd.to_datetime(cal_df.iloc[:, 1]))
        except Exception:
            trading_days = pd.DatetimeIndex([])
    else:
        trading_days = pd.DatetimeIndex([])

    if len(trading_days) == 0:
        try:
            price_df = pd.read_parquet(cfg.price_day_file)
            if not isinstance(price_df.index, pd.MultiIndex):
                price_df = price_df.set_index(["order_book_id", "date"]).sort_index()
            trading_days = pd.DatetimeIndex(
                sorted(price_df.index.get_level_values(1).unique())
            )
        except Exception:
            raise RuntimeError(
                "无法获取交易日历，请在配置中提供 trading_day_file 或可读的 price_day_file。"
            )

    weeks_sorted = pd.DatetimeIndex(sorted(weekly_port.keys()))
    results = []
    for i, d_sig in enumerate(weeks_sorted):
        d_next = weeks_sorted[i + 1] if (i + 1 < len(weeks_sorted)) else None
        if d_next is None:
            mask = (trading_days >= d_sig) & (
                trading_days <= pd.Timestamp(cfg.bt_end_date)
            )
        else:
            mask = (trading_days >= d_sig) & (trading_days < d_next)
        days_span = trading_days[mask]

        port = weekly_port[d_sig]
        if port is None or port.empty or len(days_span) == 0:
            continue

        # 找到对应的 signal_date（最近一次真正调仓周）
        eff_sig = None
        for j in range(i, -1, -1):
            d_prev_sig = weeks_sorted[j]
            is_trigger = (j % reb_k == 0)
            if is_trigger and (weekly_port[d_prev_sig] is not None) and (
                not weekly_port[d_prev_sig].empty
            ):
                eff_sig = d_prev_sig
                break
        if eff_sig is None:
            eff_sig = d_sig

        for d_day in days_span:
            for _, r in port.iterrows():
                results.append(
                    {
                        "date": d_day,
                        "stock": str(r["stock"]),
                        "weight": float(r["weight"]),
                        "score": float(r["score"]),
                        "score_neutral": float(r["score_neutral"]),
                        "score_neutral_mkt": float(r["score_neutral_mkt"]),
                        "signal_date": eff_sig,
                    }
                )

    if not results:
        raise RuntimeError("未生成任何日度持仓，请检查打分、配置或调仓频率。")

    out_df = pd.DataFrame(results).sort_values(
        ["date", "weight"], ascending=[True, False]
    )
    out_df["date"] = pd.to_datetime(out_df["date"])

    # 将“信号日”映射成“执行交易日”（下一交易日）
    cal = load_calendar(CFG.trading_day_file)
    idx = cal.values.searchsorted(out_df["date"], side="right")
    exec_dates = cal.values[idx]
    out_df["date"] = exec_dates

    pos_out = out_dir / f"positions_{cfg.run_name_out}_{weight_mode}.csv"
    ensure_dir(pos_out)
    out_df.to_csv(pos_out, index=False, float_format="%.8f")
    print(
        f"[OPT-POS] saved daily positions with rebalance_every_k={reb_k}: {pos_out}"
    )


if __name__ == "__main__":
    main()