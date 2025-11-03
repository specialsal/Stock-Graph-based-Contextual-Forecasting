# coding: utf-8
"""
optimize_position.py
生成“日度持仓”（支持低频调仓 rebalance_every_k）：
- weight_mode ∈ {"equal", "score"}
- 输入：predictions_filtered_{run_name}.parquet（或 run_name_out 版本），包含池内在各“信号周五”的打分
- 步骤：
  1) 逐“信号周五”处理：若为“调仓周”（按 rebalance_every_k），用该周打分选股+赋权；否则沿用“最近一次调仓周”的持仓与权重
  2) 用交易日历将相邻两次“信号周”之间的所有交易日，逐日复制“最近一次调仓周”的权重
  3) 输出为日度持仓 CSV：positions_{run_name_out}_{weight_mode}.csv，包含列：
     [date, stock, weight, score, score_neutral, signal_date]

注意：
- 行业中性化仅用于排序/权重；输出审计列 score_neutral
- 不做额外防御性处理，假设数据完备且日期均为交易日
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd

from backtest_rolling_config import BT_ROLL_CFG as CFG_BT
from utils import load_industry_map,load_calendar
from config import CFG


def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def _load_scores(cfg) -> pd.DataFrame:
    # 读取池内打分（信号周五）
    p = cfg.backtest_dir / f"predictions_filtered_{cfg.run_name_out}.parquet"
    if not p.exists():
        alt = cfg.backtest_dir / f"predictions_filtered_{cfg.run_name}.parquet"
        p = alt if alt.exists() else p
    df = pd.read_parquet(p)
    # 时间裁剪
    df = df[(df["date"] >= pd.Timestamp(cfg.bt_start_date)) & (df["date"] <= pd.Timestamp(cfg.bt_end_date))]
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["date", "stock", "score"])
    return df[["date", "stock", "score"]].copy()


def _winsorize_series(s: pd.Series, pct: float) -> pd.Series:
    if pct <= 0.0:
        return s
    lo = s.quantile(pct)
    hi = s.quantile(1.0 - pct)
    return s.clip(lower=lo, upper=hi)


def _neutralize_week_scores(sub: pd.DataFrame, ind_map: dict, add_intercept: bool,
                            clip_pct: float, method: str = "ols_resid") -> pd.DataFrame:
    """
    对单周（信号周五）的打分做行业中性化。返回包含 score_neutral 的 DataFrame。
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
        X = pd.get_dummies(df["ind_id"].astype(int), prefix="ind", drop_first=False)
        if add_intercept:
            X = pd.concat([pd.Series(1.0, index=X.index, name="intercept"), X], axis=1)
        y = df["score_w"].astype(float).values
        Xv = X.values.astype(float)
        beta, *_ = np.linalg.lstsq(Xv, y, rcond=None)
        yhat = Xv @ beta
        resid = y - yhat
        df["score_neutral"] = resid.astype(float)
    else:
        grp_mean = df.groupby("ind_id")["score_w"].transform("mean")
        df["score_neutral"] = (df["score_w"] - grp_mean).astype(float)

    return df.drop(columns=["ind_id", "score_w"])


def _select_and_weight(df_week: pd.DataFrame, weight_mode: str,
                       top_pct: float, max_n: int, min_n: int,
                       filter_negative_scores_long: bool,
                       col_use: str) -> pd.DataFrame:
    """
    对单周信号做选股与权重分配。返回列：stock, weight, score, score_neutral
    """
    d = df_week.sort_values([col_use, "stock"], ascending=[False, True]).copy()
    n = len(d)
    n_target = max(0, min(int(math.floor(n * top_pct)), int(max_n)))
    if n_target == 0 and n >= min_n:
        n_target = min(int(min_n), n)
    d = d.head(n_target)
    if filter_negative_scores_long:
        d = d[d[col_use] > 0]
    if len(d) < min_n:
        return pd.DataFrame(columns=["stock", "weight", "score", "score_neutral"])

    if weight_mode == "equal":
        w = np.full(len(d), 1.0 / len(d), dtype=float)
    else:
        s = d[col_use].values.astype(float)
        s = np.clip(s, 0.0, None)
        ssum = s.sum()
        w = (np.full(len(d), 1.0 / len(d), dtype=float) if ssum <= 0 else s / ssum)
    d = d.assign(weight=w)
    return d[["stock", "weight", "score", "score_neutral"]]


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

    # 行业中性化配置
    do_neutral = bool(getattr(cfg, "neutralize_enable", False))
    neutral_method = str(getattr(cfg, "neutralize_method", "ols_resid")).lower()
    add_intercept = bool(getattr(cfg, "neutralize_add_intercept", True))
    clip_pct = float(getattr(cfg, "neutralize_clip_pct", 0.0))
    ind_map = load_industry_map(cfg.industry_map_file) if do_neutral else {}

    # 准备“周度组合（含低频调仓逻辑）”
    weekly_port = {}  # date -> DataFrame(stock, weight, score, score_neutral)
    last_reb_sig_date = None
    last_reb_port = None

    for i, d in enumerate(weeks):
        sub = scores_all[scores_all["date"] == d].copy()

        # 是否触发调仓
        trigger_reb = (i % reb_k == 0)

        if trigger_reb:
            # 计算当周权重
            if do_neutral:
                sub_n = _neutralize_week_scores(sub, ind_map, add_intercept, clip_pct, neutral_method)
                col_use = "score_neutral" if sub_n["score_neutral"].notna().any() else "score"
            else:
                sub_n = sub.copy()
                sub_n["score_neutral"] = sub_n["score"]
                col_use = "score"

            dsel = _select_and_weight(
                df_week=sub_n.assign(score=sub_n["score"].astype(float)),
                weight_mode=weight_mode,
                top_pct=float(cfg.top_pct),
                max_n=int(cfg.max_n_stocks),
                min_n=int(cfg.min_n_stocks),
                filter_negative_scores_long=bool(cfg.filter_negative_scores_long),
                col_use=col_use
            )
            if not dsel.empty:
                # 成功建仓：记录为当前有效调仓组合
                weekly_port[pd.Timestamp(d)] = dsel
                last_reb_sig_date = pd.Timestamp(d)
                last_reb_port = dsel
            else:
                # 本应调仓但不足以建仓：沿用上一有效组合（若有）
                if last_reb_port is not None:
                    weekly_port[pd.Timestamp(d)] = last_reb_port.copy()
                else:
                    weekly_port[pd.Timestamp(d)] = pd.DataFrame(columns=["stock", "weight", "score", "score_neutral"])
        else:
            # 非调仓周：沿用最近一次调仓组合
            if last_reb_port is not None:
                weekly_port[pd.Timestamp(d)] = last_reb_port.copy()
            else:
                # 尚无历史有效组合
                if do_neutral:
                    sub_n = _neutralize_week_scores(sub, ind_map, add_intercept, clip_pct, neutral_method)
                else:
                    sub_n = sub.assign(score_neutral=sub["score"])
                weekly_port[pd.Timestamp(d)] = pd.DataFrame(columns=["stock", "weight", "score", "score_neutral"])

    # 将周度组合展开为“日度持仓”
    # 规则：在相邻两信号周之间（含起始信号日，不含下一信号日），对每个交易日复制“最近一次调仓周”的权重
    # 构造全交易日序列：从 scores 的 min/max 直接铺，以 positions 的日期集合为准
    # 若你有交易日文件，可在此接入；这里按最小实现，用周期间隔日历推断
    all_days = pd.DatetimeIndex(sorted(scores_all["date"].unique()))
    # 上面是周五序列；为了生成日度，需要交易日日历。若项目已有 trading_day_file，优先使用：
    if hasattr(cfg, "trading_day_file"):
        try:
            cal_df = pd.read_csv(cfg.trading_day_file, header=None, skiprows=1)
            trading_days = pd.DatetimeIndex(pd.to_datetime(cal_df.iloc[:, 1]))
        except Exception:
            trading_days = pd.DatetimeIndex([])
    else:
        trading_days = pd.DatetimeIndex([])

    if len(trading_days) == 0:
        # 回退：用价格文件推断
        try:
            price_df = pd.read_parquet(cfg.price_day_file)
            if not isinstance(price_df.index, pd.MultiIndex):
                price_df = price_df.set_index(["order_book_id", "date"]).sort_index()
            trading_days = pd.DatetimeIndex(sorted(price_df.index.get_level_values(1).unique()))
        except Exception:
            raise RuntimeError("无法获取交易日历，请在配置中提供 trading_day_file 或可读的 price_day_file。")

    weeks_sorted = pd.DatetimeIndex(sorted(weekly_port.keys()))
    results = []
    for i, d_sig in enumerate(weeks_sorted):
        d_next = weeks_sorted[i + 1] if (i + 1 < len(weeks_sorted)) else None
        # 该信号周期的交易日范围
        if d_next is None:
            mask = (trading_days >= d_sig) & (trading_days <= pd.Timestamp(cfg.bt_end_date))
        else:
            mask = (trading_days >= d_sig) & (trading_days < d_next)
        days_span = trading_days[mask]

        # 对应组合（注意：weekly_port[d_sig] 可能是沿用的“上次调仓组合”）
        port = weekly_port[d_sig]
        if port is None or port.empty or len(days_span) == 0:
            continue

        # 确定该周对应的“signal_date”：应为“最近一次触发调仓的周五”
        # 我们需要追溯到该周记录中沿用的来源。策略：在 weekly_port 映射中，连续相同对象意味着沿用；
        # 这里简化处理：对每个 d_sig，若它自身是触发周（i % reb_k == 0 且当周成功建仓），signal_date = d_sig；
        # 若非触发周或当周建仓失败，则 signal_date = 最近一个 <= d_sig 且 i%k==0 且成功建仓的周五
        # 为实现该追溯，预先构造一个 map: effective_signal_date_for_week
        # 简化：在上面循环已维护 last_reb_sig_date，因此可以在这里再做一次前向遍历来赋值。
        # 为避免二次遍历，我们在此动态回溯：
        # 找到从当前 d_sig 往前第 j 个使 (idx % reb_k == 0) 且 weekly_port[that_day].shape>0 的周五
        eff_sig = None
        # 向前回溯
        for j in range(i, -1, -1):
            d_prev_sig = weeks_sorted[j]
            is_trigger = (j % reb_k == 0)
            if is_trigger and (weekly_port[d_prev_sig] is not None) and (not weekly_port[d_prev_sig].empty):
                eff_sig = d_prev_sig
                break
        if eff_sig is None:
            eff_sig = d_sig  # 兜底

        # 复制到每日
        for d_day in days_span:
            for _, r in port.iterrows():
                results.append({
                    "date": d_day,
                    "stock": str(r["stock"]),
                    "weight": float(r["weight"]),
                    "score": float(r["score"]),
                    "score_neutral": float(r["score_neutral"]),
                    "signal_date": eff_sig
                })

    if not results:
        raise RuntimeError("未生成任何日度持仓，请检查打分、配置或调仓频率。")

    out_df = pd.DataFrame(results).sort_values(["date", "weight"], ascending=[True, False])
    out_df["date"] = pd.to_datetime(out_df["date"])
    cal = load_calendar(CFG.trading_day_file)
    # 用 searchsorted 找到“严格大于信号日”的下一个交易日索引（向右）
    idx = cal.values.searchsorted(out_df['date'], side="right")
    # 批量替换为执行日
    exec_dates = cal.values[idx]
    out_df["date"] = exec_dates
    pos_out = out_dir / f"positions_{cfg.run_name_out}_{weight_mode}.csv"
    ensure_dir(pos_out)
    out_df.to_csv(pos_out, index=False, float_format="%.8f")
    print(f"[OPT-POS] saved daily positions with rebalance_every_k={reb_k}: {pos_out}")


if __name__ == "__main__":
    main()