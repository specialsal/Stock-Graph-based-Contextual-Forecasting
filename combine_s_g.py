# coding: utf-8
"""
combine_s_g.py
组合器：你的股票策略(S) + 黄金ETF(G) 的环境自适应配置（周频）

特点
- 读取主策略周频 nav_{run}.csv（含 ret_total / nav）。
- 读取黄金ETF日线（parquet 或 csv），合成周收益（O2C 或 C2C 可选）。
- 构造 env_strength（波动分位 + 广度 + 近4周方向）。
- 动态分配 S/G 权重（线性门控或三段式状态机），带平滑和每周变动上限。
- 即使某周没有可用收益，也输出当周权重与环境；收益列保留 NaN；NAV 将 NaN 视为0收益以连贯。
- 导出组合 NAV/权重/环境/指标与图。

使用
- 根据实际路径与偏好修改 CONFIG 顶部参数，运行：python combine_s_g.py
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import math
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from backtest_rolling_config import BT_ROLL_CFG


# ====================== 用户参数 ======================
@dataclass
class CONFIG:
    # 运行名（用于默认路径拼接）
    run_name: str = "tr1gat1win50"

    # 输入路径
    main_nav_path: Path = Path(f"./backtest_rolling/tr1gat1win50/nav_tr1gat1win50.csv")
    gold_day_path: Path = Path("./data/raw/gold_etf_day.parquet")  # 若不存在将尝试同名 .csv
    trading_day_csv: Path = Path("./data/raw/trading_day.csv")
    index_day_path: Path = Path("./data/raw/index_price_day.parquet")  # 用于环境分数（沪深300作市场代理），若缺失则关闭方向修正

    # 输出目录
    out_dir: Path = Path(f"./backtest_rolling/tr1gat1win50/s_g_combo")

    # 黄金周收益合成口径
    gold_o2c: bool = True  # True=O2C；False=C2C

    # 环境分数 env_strength 参数
    env_vol_low: int = 1
    env_vol_high: int = 5
    breadth_wide_penalty: float = 0
    breadth_narrow_penalty: float = 0
    dir_lookback_weeks: int = 3
    dir_up_thresh: float = 0.02   # 近3周 > +2%
    dir_dn_thresh: float = -0.02  # 近3周 < -2%
    dir_soften_mult: float = 0.5  # 慢牛时 ×0.5
    dir_harden_add: float = 0.1   # 急跌时 +0.1
    # w_G = w_G_min + (w_G_max - w_G_min) * env_strength

    # 权重调度模式（"linear" 或 "state"）
    alloc_mode: str = "linear"
    # 线性门控
    wG_min: float = 0.10
    wG_max: float = 0.90
    # 三段式状态机阈值与权重
    state_low: float = 0.3
    state_high: float = 0.6
    wG_attack: float = 0.05
    wG_neutral_lo_hi: Tuple[float, float] = (0.20, 0.35)
    wG_defense: float = 0.50
    hysteresis_weeks: int = 2  # 连续满足阈值周数才切换状态

    # 权重平滑与变动上限
    weight_smooth_lambda: float = 0.3    # w_t = (1-λ)*w_target + λ*w_{t-1}
    weekly_change_cap: float = 0.10      # |Δw_G| ≤ 10%/周

    # 黄金ETF筛选（文件如含多资产）
    gold_code: Optional[str] = None  # None 表示文件只含一个资产

    # 指标频率
    annual_freq: int = 52

    # 绘图范围（可选）
    plot_start: Optional[str] = '2017-02-10'  # 例如 "2017-01-01"
    plot_end: Optional[str] = None

CFG = CONFIG()
# =====================================================


# ====================== 工具函数 ======================
def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def read_nav_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"未找到主策略NAV：{path}")
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date")
    df = df.replace([np.inf, -np.inf], np.nan).sort_index()
    return df

def load_trading_days(p: Path) -> pd.DatetimeIndex:
    if not p.exists():
        raise FileNotFoundError(f"未找到交易日历：{p}")
    df = pd.read_csv(p)
    col = "trading_day" if "trading_day" in df.columns else df.columns[0]
    days = pd.to_datetime(df[col], errors="coerce").dropna().sort_values().unique()
    return pd.DatetimeIndex(days)

def weekly_fridays_from_days(days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    days = pd.DatetimeIndex(days).sort_values()
    week_end: List[pd.Timestamp] = []
    for i in range(len(days) - 1):
        if (days[i + 1] - days[i]).days > 1:
            week_end.append(days[i])
    if len(days) > 0:
        if not week_end or week_end[-1] != days[-1]:
            week_end.append(days[-1])
    return pd.DatetimeIndex(week_end)

def compute_weekly_ret_from_day(day_df: pd.DataFrame,
                                fridays: pd.DatetimeIndex,
                                mode_o2c: bool = True) -> pd.Series:
    """
    day_df: index=date, cols至少包含 open, close
    返回：周收益（索引=周五锚点 f0），O2C 或 C2C
    """
    day_df = day_df.sort_index()
    avail = day_df.index
    rets = {}
    for i in range(len(fridays) - 1):
        f0, f1 = fridays[i], fridays[i + 1]
        i0 = avail[avail <= f0]; i1 = avail[avail <= f1]
        if len(i0) == 0 or len(i1) == 0:
            continue
        d0, d1 = i0[-1], i1[-1]
        if mode_o2c:
            # 开在 d0 的下一交易日，收在 d1
            pos = avail.get_indexer([d0])[0]
            if pos < 0 or pos + 1 >= len(avail):
                continue
            start = avail[pos + 1]
            if start > d1:
                # 假期或停牌导致下一交易日跨过 d1，无法构造
                continue
            if ("open" not in day_df.columns) or ("close" not in day_df.columns):
                continue
            o = float(day_df.loc[start, "open"])
            c = float(day_df.loc[d1, "close"])
        else:
            # C2C：d0收 -> d1收
            if ("close" not in day_df.columns):
                continue
            o = float(day_df.loc[d0, "close"])
            c = float(day_df.loc[d1, "close"])
        if o <= 0 or not np.isfinite(o) or not np.isfinite(c):
            continue
        rets[f0] = c / o - 1.0
    return pd.Series(rets).sort_index()

def qlabel_from_rank(rank_series: pd.Series, labels5) -> pd.Series:
    # 将 [0,1] 的滚动分位映射到 5 档标签
    s = rank_series.clip(0, 1).copy()
    # 为防止样本过少 qcut 抛错，添加微小噪声
    if s.notna().sum() >= 20:
        try:
            return pd.qcut(s, 5, labels=labels5)
        except Exception:
            pass
    # 退化：按阈值切
    bins = [0.2, 0.4, 0.6, 0.8, 1.01]
    out = pd.Series(index=s.index, dtype=object)
    for i, b in enumerate(bins):
        mask = (s <= b) & (out.isna())
        out.loc[mask] = labels5[i]
    return out

def calc_env_strength(fridays: pd.DatetimeIndex,
                      mkt_daily_close: pd.Series,
                      up_ratio_daily: pd.Series,
                      cfg: CONFIG) -> pd.Series:
    """
    使用市场日收盘近似：
    - 波动：20日波动的滚动252日分位，映射到 vol_q ∈ {1..5}
    - 广度：上涨占比的滚动252日分位，映射到 Q1..Q5，Q1/Q5 罚分
    - 方向：近4周累计收益>阈值则软化，<阈值则加防御
    返回：索引=fridays ∩ mkt日期范围 的 env_strength
    """
    if mkt_daily_close is None or len(mkt_daily_close) == 0:
        # 无市场数据则给空序列，由上层填充
        return pd.Series(index=fridays, dtype=float)

    mkt_daily_close = mkt_daily_close.sort_index()
    ret_d = mkt_daily_close.pct_change().fillna(0.0)

    vol20 = ret_d.rolling(20).std() * np.sqrt(252)
    vol_rank = vol20.rolling(252, min_periods=60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan,
        raw=False
    )
    vol_rank = vol_rank.clip(0, 1).ffill()
    # 将分位映射为1..5
    vol_q_lab = qlabel_from_rank(vol_rank, labels5=[1, 2, 3, 4, 5]).astype(float)

    # breadth
    if up_ratio_daily is None or len(up_ratio_daily) == 0:
        up_ratio_daily = pd.Series(0.5, index=ret_d.index)
    brank = up_ratio_daily.rolling(252, min_periods=60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan,
        raw=False
    ).clip(0, 1).ffill()
    breadth_lab = qlabel_from_rank(brank, labels5=["Q1(窄)", "2", "3", "4", "Q5(宽)"])

    # 对齐到周五
    env_rows = []
    for d in fridays:
        ok = mkt_daily_close.index[mkt_daily_close.index <= d]
        if len(ok) == 0:
            continue
        dd = ok[-1]
        vq = float(vol_q_lab.loc[dd]) if dd in vol_q_lab.index and pd.notna(vol_q_lab.loc[dd]) else 3.0
        bl = str(breadth_lab.loc[dd]) if dd in breadth_lab.index and pd.notna(breadth_lab.loc[dd]) else "3"

        # base：vol_q -> env_strength
        low, high = cfg.env_vol_low, cfg.env_vol_high
        if vq <= low:
            env = 1.0
        elif vq >= high:
            env = 0.0
        else:
            env = (high - vq) / max(1.0, (high - low))

        # breadth 罚分
        if bl == "Q5(宽)":
            env = min(1.0, env + cfg.breadth_wide_penalty)
        elif bl == "Q1(窄)":
            env = min(1.0, env + cfg.breadth_narrow_penalty)

        env_rows.append({"date": d, "env_base": env})
    env_df = pd.DataFrame(env_rows).set_index("date")

    # 方向修正：近4周累计收益（C2C）
    avail = mkt_daily_close.index
    week_ret = []
    for i in range(len(fridays) - 1):
        f0, f1 = fridays[i], fridays[i + 1]
        i0 = avail[avail <= f0]; i1 = avail[avail <= f1]
        if len(i0) == 0 or len(i1) == 0:
            continue
        d0, d1 = i0[-1], i1[-1]
        r = float(mkt_daily_close.loc[d1] / mkt_daily_close.loc[d0] - 1.0)
        week_ret.append((f0, r))
    wr = pd.Series({d: r for d, r in week_ret}).sort_index()

    env_df["env_strength"] = env_df["env_base"].values
    cum4 = wr.rolling(cfg.dir_lookback_weeks).sum()
    for d in env_df.index:
        if d in cum4.index and pd.notna(cum4.loc[d]):
            x = float(cum4.loc[d])
            if x > cfg.dir_up_thresh:
                env_df.loc[d, "env_strength"] = env_df.loc[d, "env_strength"] * cfg.dir_soften_mult
            elif x < cfg.dir_dn_thresh:
                env_df.loc[d, "env_strength"] = min(1.0, env_df.loc[d, "env_strength"] + cfg.dir_harden_add)
    env_df["env_strength"] = env_df["env_strength"].clip(0, 1)
    return env_df["env_strength"]

def calc_stats(rets: pd.Series, annual_freq: int = 52) -> Dict[str, float]:
    r = pd.Series(rets).dropna().astype(float).values
    if len(r) == 0:
        return {"total_return": 0.0, "annual_return": 0.0, "annual_vol": 0.0,
                "sharpe": float("nan"), "max_drawdown": 0.0, "calmar": float("nan"), "n_periods": 0}
    nav = np.cumprod(1.0 + r)
    total = float(nav[-1] - 1.0)
    ann = float((1.0 + total) ** (annual_freq / max(1, len(r))) - 1.0)
    vol = float(np.std(r, ddof=1)) * math.sqrt(annual_freq) if len(r) > 1 else 0.0
    sharpe = float(np.mean(r) / (np.std(r, ddof=1) if len(r) > 1 else np.nan)) * math.sqrt(annual_freq)
    peak = np.maximum.accumulate(nav)
    dd = 1.0 - nav / peak
    maxdd = float(np.nanmax(dd))
    calmar = float(ann / maxdd) if maxdd > 1e-12 else float("nan")
    return {"total_return": total, "annual_return": ann, "annual_vol": vol,
            "sharpe": sharpe, "max_drawdown": maxdd, "calmar": calmar, "n_periods": int(len(r))}

def read_parquet_or_csv(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet" and path.exists():
        return pd.read_parquet(path)
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"未找到 {path} 或 {csv_path}")

def pick_index_close_from_parquet(df: pd.DataFrame) -> Optional[pd.Series]:
    """从 index_price_day.parquet 中挑选沪深300或首列的收盘价"""
    try:
        if "datetime" in df.columns and "order_book_id" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
            df = df.dropna(subset=["datetime"])
            close_piv = df.pivot_table(index="datetime", columns="order_book_id", values="close")
        elif isinstance(df.index, pd.MultiIndex) and "close" in df.columns:
            close_piv = df["close"].unstack(0)
        else:
            # 如果已经是宽表
            col_close = [c for c in df.columns if str(c).lower().endswith("close")]
            if len(col_close) == 1:
                s = df[col_close[0]]
                s.index = pd.to_datetime(df.index, errors="coerce")
                return s.sort_index()
            elif "close" in df.columns:
                s = df["close"]
                s.index = pd.to_datetime(df.index, errors="coerce")
                return s.sort_index()
            return None
        # 选沪深300或第一列
        cols = close_piv.columns.astype(str)
        pick = None
        for pat in ["000300", "CSI300", "HS300"]:
            match = [c for c in cols if pat in c]
            if match:
                pick = match[0]; break
        if pick is None and len(close_piv.columns) > 0:
            pick = close_piv.columns[0]
        if pick is None:
            return None
        s = close_piv[pick].dropna()
        s.index = pd.to_datetime(s.index, errors="coerce")
        return s.sort_index()
    except Exception:
        return None


# ====================== 主流程 ======================
def main(cfg: CONFIG = CFG):
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    # 读策略S
    dfS = read_nav_csv(cfg.main_nav_path)
    if "ret_total" not in dfS.columns or "nav" not in dfS.columns:
        raise RuntimeError("主策略 nav 文件需包含列：ret_total 和 nav")
    ret_S = dfS["ret_total"].astype(float).copy()
    # 主策略日期范围
    s_start, s_end = ret_S.index.min(), ret_S.index.max()

    # 交易日/周五
    days = load_trading_days(cfg.trading_day_csv)
    fridays_all = weekly_fridays_from_days(days)
    # 限定到策略可见范围稍扩展一周尾
    fridays = fridays_all[(fridays_all >= s_start - pd.Timedelta(days=7)) & (fridays_all <= s_end)]

    # 读黄金ETF日线
    gday = read_parquet_or_csv(cfg.gold_day_path)
    # 标准化列
    if "datetime" in gday.columns:
        gday["date"] = pd.to_datetime(gday["datetime"], errors="coerce")
    elif "date" in gday.columns:
        gday["date"] = pd.to_datetime(gday["date"], errors="coerce")
    else:
        # 尝试索引
        if gday.index.name is not None:
            gday = gday.reset_index().rename(columns={gday.index.name: "date"})
            gday["date"] = pd.to_datetime(gday["date"], errors="coerce")
        else:
            raise RuntimeError("黄金ETF日线缺少 date/datetime 列，且无法从索引推断")
    if cfg.gold_code and "order_book_id" in gday.columns:
        gday = gday[gday["order_book_id"] == cfg.gold_code].copy()
    need_cols = ["open", "close"]
    for c in need_cols:
        if c not in gday.columns:
            raise RuntimeError(f"黄金ETF日线缺少列：{c}")
    gday = gday.dropna(subset=["date"]).sort_values("date")
    gday = gday[["date", "open", "close"]].dropna()
    gday = gday.set_index("date").sort_index()

    # 黄金周收益（全范围），稍微扩展以保证覆盖 s_end
    fridays_for_gold = fridays_all[(fridays_all >= gday.index.min())]
    ret_G_full = compute_weekly_ret_from_day(gday, fridays_for_gold, mode_o2c=cfg.gold_o2c).shift(-1)

    # 对齐：使用并集索引，允许一侧缺失（你的要求）
    idx_union = ret_S.index.union(ret_G_full.index).sort_values()
    idx_union = idx_union[idx_union>=s_start]
    ret_S = ret_S.reindex(idx_union)
    ret_G = ret_G_full.reindex(idx_union)

    # 环境：市场代理（沪深300）
    disable_dir = False
    mkt_proxy_close = None
    try:
        idx_df = read_parquet_or_csv(cfg.index_day_path)
        mkt_proxy_close = pick_index_close_from_parquet(idx_df)
        if mkt_proxy_close is None or len(mkt_proxy_close) == 0:
            disable_dir = True
            warnings.warn("未能从 index_day_path 提取市场收盘价，方向修正将关闭。")
    except FileNotFoundError:
        disable_dir = True
        warnings.warn("未找到 index_day_path，方向修正将关闭。")

    # 广度近似
    if mkt_proxy_close is not None and len(mkt_proxy_close) > 0:
        r = mkt_proxy_close.pct_change().fillna(0.0)
        up_ratio_daily = (r > 0).astype(float)
    else:
        up_ratio_daily = pd.Series(0.5, index=days)

    # 计算 env_strength（按周五锚点）
    env_fridays = idx_union  # 需要在所有周上给出环境
    if disable_dir:
        # 方向阈值设极端以关闭方向修正
        tmp_cfg = CONFIG(**CFG.__dict__)
        tmp_cfg.dir_up_thresh = 999
        tmp_cfg.dir_dn_thresh = -999
        env = calc_env_strength(fridays=env_fridays,
                                mkt_daily_close=mkt_proxy_close if mkt_proxy_close is not None else pd.Series(index=days, dtype=float),
                                up_ratio_daily=up_ratio_daily,
                                cfg=tmp_cfg)
    else:
        env = calc_env_strength(fridays=env_fridays,
                                mkt_daily_close=mkt_proxy_close,
                                up_ratio_daily=up_ratio_daily,
                                cfg=CFG)

    # env 用于权重演化的版本：缺失用前值，否则0.5
    env_for_weight = env.copy().ffill().fillna(0.5)

    # 权重调度（每周都有）
    wG_vals: List[float] = []
    state = "neutral"
    hold_counter = 0
    prev_wG = CFG.wG_min
    for d in idx_union:
        e = float(env_for_weight.loc[d]) if d in env_for_weight.index and pd.notna(env_for_weight.loc[d]) else 0.5
        if CFG.alloc_mode == "linear":
            target = CFG.wG_min + (CFG.wG_max - CFG.wG_min) * e
        else:
            # 三段式状态机（带滞后）
            if state == "attack":
                target = CFG.wG_attack
                if e >= CFG.state_high:
                    hold_counter += 1
                    if hold_counter >= CFG.hysteresis_weeks:
                        state, hold_counter = "defense", 0
                elif e > CFG.state_low:
                    state, hold_counter = "neutral", 0
            elif state == "defense":
                target = CFG.wG_defense
                if e <= CFG.state_low:
                    hold_counter += 1
                    if hold_counter >= CFG.hysteresis_weeks:
                        state, hold_counter = "attack", 0
                elif e < CFG.state_high:
                    state, hold_counter = "neutral", 0
            else:
                # neutral：在低高之间插值
                lo, hi = CFG.wG_neutral_lo_hi
                if CFG.state_high == CFG.state_low:
                    target = (lo + hi) / 2.0
                else:
                    target = float(np.interp(e, [CFG.state_low, CFG.state_high], [lo, hi]))
                if e <= CFG.state_low:
                    hold_counter += 1
                    if hold_counter >= CFG.hysteresis_weeks:
                        state, hold_counter = "attack", 0
                elif e >= CFG.state_high:
                    hold_counter += 1
                    if hold_counter >= CFG.hysteresis_weeks:
                        state, hold_counter = "defense", 0
                else:
                    hold_counter = 0
        # 平滑
        w_t = (1.0 - CFG.weight_smooth_lambda) * target + CFG.weight_smooth_lambda * prev_wG
        # 变动上限
        w_t = float(np.clip(w_t, prev_wG - CFG.weekly_change_cap, prev_wG + CFG.weekly_change_cap))
        w_t = float(np.clip(w_t, 0.0, 1.0))
        wG_vals.append(w_t)
        prev_wG = w_t

    wG = pd.Series(wG_vals, index=idx_union)
    wS = 1.0 - wG

    # 组合收益（允许 NaN）
    cost = (BT_ROLL_CFG.slippage_bps + BT_ROLL_CFG.fee_bps) * 1e-4
    ret_combo = (wS.shift(1) * ret_S.fillna(0.0)) + (wG.shift(1) * ret_G*(1-cost))
    nav_S = (1.0 + ret_S.fillna(0.0)).cumprod()
    nav_G = (1.0 + ret_G.fillna(0.0)).cumprod()
    nav_combo = (1.0 + ret_combo.fillna(0.0)).cumprod()

    # 汇总输出
    out = pd.DataFrame({
        "ret_S": ret_S.fillna(0.0),
        "ret_G": ret_G,
        "ret_combo": ret_combo.fillna(0.0),
        "nav_S": nav_S,
        "nav_G": nav_G,
        "nav_combo": nav_combo,
        "w_S": wS,
        "w_G": wG,
        "env_strength": env.reindex(idx_union)
    })

    # 可选：绘图范围裁剪
    if CFG.plot_start or CFG.plot_end:
        mask = pd.Series(True, index=out.index)
        if CFG.plot_start:
            mask &= out.index >= pd.to_datetime(CFG.plot_start)
        if CFG.plot_end:
            mask &= out.index <= pd.to_datetime(CFG.plot_end)
        out_plot = out.loc[mask]
    else:
        out_plot = out

    # 保存
    out_path = cfg.out_dir / "combo_nav.csv"
    ensure_dir(out_path)
    out.rename(columns={'nav_combo':'nav', 'ret_combo':'ret_total'}).to_csv(out_path, float_format="%.8f")
    print(f"[COMBO] saved {out_path}")

    # 指标
    mS = calc_stats(out["ret_S"], annual_freq=CFG.annual_freq)
    mG = calc_stats(out["ret_G"], annual_freq=CFG.annual_freq)
    mC = calc_stats(out["ret_combo"], annual_freq=CFG.annual_freq)
    metrics = {"S": mS, "G": mG, "Combo": mC}
    with open(cfg.out_dir / "metrics_combo.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 年度指标（按组合）
    tmp = out.copy()
    tmp["year"] = tmp.index.year
    by_year = []
    for y, g in tmp.groupby("year"):
        my = calc_stats(g["ret_combo"], annual_freq=CFG.annual_freq)
        by_year.append({"year": int(y), **my})
    pd.DataFrame(by_year).to_csv(cfg.out_dir / "metrics_by_year_combo.csv", index=False)

    # 图：NAV
    fig, ax = plt.subplots(1, 1, figsize=(11, 4.6))
    ax.plot(out_plot.index, out_plot["nav_S"], label="Strategy S", lw=1.4)
    ax.plot(out_plot.index, out_plot["nav_G"], label="Gold ETF G", lw=1.4)
    ax.plot(out_plot.index, out_plot["nav_combo"], label="Combo", lw=2.0)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("NAV: S / G / Combo")
    ax.set_ylabel("NAV")
    fig.tight_layout()
    fig.savefig(cfg.out_dir / "nav_combo.png", dpi=160)
    plt.close(fig)

    # 图：权重与环境
    fig, ax = plt.subplots(2, 1, figsize=(11, 6.0), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax[0].plot(out_plot.index, out_plot["w_G"], label="w_G", lw=1.6, color="#cc9900")
    ax[0].plot(out_plot.index, out_plot["w_S"], label="w_S", lw=1.0, color="#2b8cbe", alpha=0.8)
    ax[0].set_ylim(0, 1)
    ax[0].grid(alpha=0.3)
    ax[0].legend(loc="upper right")
    ax[0].set_title("Weights (G and S)")
    ax[1].plot(out_plot.index, out_plot["env_strength"], color="#555555", lw=1.2)
    ax[1].set_ylim(0, 1)
    ax[1].grid(alpha=0.3)
    ax[1].set_ylabel("env_strength")
    fig.tight_layout()
    fig.savefig(cfg.out_dir / "weights_combo.png", dpi=160)
    plt.close(fig)

    # 调试信息
    print("[INFO] Last dates:")
    print("  ret_S last:", pd.to_datetime(out['ret_S'].dropna().index.max()) if out['ret_S'].notna().any() else None)
    print("  ret_G last:", pd.to_datetime(out['ret_G'].dropna().index.max()) if out['ret_G'].notna().any() else None)
    print("  ret_combo last (non-NaN):", pd.to_datetime(out['ret_combo'].dropna().index.max()) if out['ret_combo'].notna().any() else None)
    print("  out last index:", out.index.max())
    miss = out.index[out["ret_S"].isna() | out["ret_G"].isna()]
    if len(miss) > 0:
        print("  Weeks with missing returns (either S or G):", miss[-10:].tolist())
    print(f"[COMBO] Metrics Combo: {metrics['Combo']}")

if __name__ == "__main__":
    main()