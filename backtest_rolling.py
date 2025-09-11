# coding: utf-8
"""
滚动回测：在每个训练窗口结束时（pred_date）使用该窗口“验证集最优模型”
对后续 step_weeks 的周五进行预测与交易；若尾部不足 step_weeks，则尽量使用能覆盖的部分。
将各窗口的预测片段顺序拼接，生成总体回测曲线。
输出：
- predictions_filtered_{run_name}.parquet：合并后的“池内打分”记录（含无法计算收益的尾周）
- nav_{run_name}.csv / nav_{run_name}.png：总体净值与图
- nav_marked_{run_name}.png：在净值图上用半透明背景区分各窗口预测片段
- metrics_{run_name}.json：总体指标（与 backtest.py 一致）
- metrics_by_window_{run_name}.csv：窗口级指标（便于诊断）
实现细节尽量复用 backtest.py（过滤/标准化/组合构建/指标计算），保持一致的口径。
"""

import math
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import h5py
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from backtest_rolling_config import BT_ROLL_CFG
from config import CFG  # 读取训练参数中的 step_weeks 等
from model import GCFNet
from utils import load_calendar, weekly_fridays, load_industry_map
from train_utils import make_filter_fn, load_stock_info, load_flag_table  # 复用筛选逻辑

plt.switch_backend("Agg")  # 无界面环境保存图


def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def list_h5_groups(h5: h5py.File) -> pd.DataFrame:
    rows = []
    for k in h5.keys():
        if not k.startswith("date_"):
            continue
        d = h5[k].attrs.get("date", None)
        if d is None:
            continue
        if isinstance(d, bytes):
            d = d.decode("utf-8")
        rows.append({"group": k, "date": pd.to_datetime(d)})
    if not rows:
        return pd.DataFrame(columns=["group", "date"])
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


def date_to_group(h5: h5py.File) -> Dict[pd.Timestamp, str]:
    df = list_h5_groups(h5)
    return {row["date"]: row["group"] for _, row in df.iterrows()}


def infer_factor_dim(h5: h5py.File) -> int:
    if "factor_cols" in h5.attrs:
        return int(len(h5.attrs["factor_cols"]))
    # fallback: look at first group
    for k in h5.keys():
        if "factor" in h5[k]:
            arr = np.asarray(h5[k]["factor"][:])
            arr = np.squeeze(arr)
            return int(arr.shape[-1])
    raise RuntimeError("无法推断因子维度")


class ScalerPayload:
    """
    负责承载并应用训练期的标准化参数（若存在）。
    mean/std 形状在训练端通常为 [1,1,C]；这里兼容 [C]、[1,C]、[1,1,C]。
    """
    def __init__(self, mean: Optional[np.ndarray], std: Optional[np.ndarray]):
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32)
        self.std  = None if std  is None else np.asarray(std,  dtype=np.float32)

        if self.mean is not None and self.std is not None:
            # 统一成 [1,1,C]
            if self.mean.ndim == 1:
                self.mean = self.mean.reshape(1, 1, -1)
            elif self.mean.ndim == 2:
                self.mean = self.mean.reshape(1, self.mean.shape[0], self.mean.shape[1]).mean(axis=1, keepdims=False).reshape(1, 1, -1)
            elif self.mean.ndim == 3:
                pass
            else:
                raise ValueError(f"scaler_mean 维度不支持: {self.mean.shape}")

            if self.std.ndim == 1:
                self.std = self.std.reshape(1, 1, -1)
            elif self.std.ndim == 2:
                self.std = self.std.reshape(1, self.std.shape[0], self.std.shape[1]).mean(axis=1, keepdims=False).reshape(1, 1, -1)
            elif self.std.ndim == 3:
                pass
            else:
                raise ValueError(f"scaler_std 维度不支持: {self.std.shape}")

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        X: [N,T,C] 或 [N,T,1,C]（将被 squeeze 到 [N,T,C]）
        返回与训练一致的标准化结果；若无 mean/std 则原样返回。
        """
        X = np.squeeze(np.asarray(X))
        if self.mean is None or self.std is None:
            return X.astype(np.float32)
        return ((X - self.mean) / (self.std + 1e-6)).astype(np.float32)


def compute_weekly_returns(close_pivot: pd.DataFrame,
                           fridays: pd.DatetimeIndex) -> Tuple[Dict[pd.Timestamp, pd.Series], Dict[pd.Timestamp, pd.Timestamp]]:
    """
    对每个周五 f，找 <=f 的最后交易日 d0，下一周五同理得到 d1，计算 (close[d1]/close[d0]-1)
    返回：
      - weekly_ret: {friday -> Series(index=stock) 的一周收益}
      - friday_start_map: {friday -> d0} 起点交易日映射（用于过滤时的口径日）
    """
    avail = close_pivot.index
    out = {}
    start_date_map = {}
    for i in range(len(fridays) - 1):
        f0 = fridays[i]
        f1 = fridays[i+1]
        i0 = avail[avail <= f0]
        i1 = avail[avail <= f1]
        if len(i0) == 0 or len(i1) == 0:
            continue
        d0 = i0[-1]; d1 = i1[-1]
        if d0 >= d1:
            continue
        start = close_pivot.loc[d0]
        end = close_pivot.loc[d1]
        ret = (end / start - 1).replace([np.inf, -np.inf], np.nan)
        out[f0] = ret
        start_date_map[f0] = d0
    return out, start_date_map


def build_portfolio(scores: pd.DataFrame,
                    weekly_ret: Dict[pd.Timestamp, pd.Series],
                    mode: str = "ls",
                    top_pct: float = 0.2,
                    bottom_pct: float = 0.2,
                    min_n: int = 20,
                    max_n: int = 200,
                    long_w: float = 1.0,
                    short_w: float = 1.0,
                    slippage_bps: float = 0.0,
                    fee_bps: float = 0.0) -> pd.DataFrame:
    """
    输入每周打分 + 每周实际收益，输出周频净值序列
    """
    if scores.empty:
        return pd.DataFrame(columns=["date", "ret_total"]).set_index("date")

    dates = sorted(set(scores["date"]))
    records = []

    for d in dates:
        df_d = scores[scores["date"] == d].copy()
        if d not in weekly_ret:
            continue
        ret_s = weekly_ret[d]  # Series(stock -> ret)

        df_d = df_d.merge(ret_s.rename("ret"), left_on="stock", right_index=True, how="inner")
        df_d = df_d.dropna(subset=["ret", "score"])
        if len(df_d) < min_n:
            records.append({"date": d, "ret_long": 0.0, "ret_short": 0.0, "ret_total": 0.0, "n_long": 0, "n_short": 0})
            continue

        df_d = df_d.sort_values("score", ascending=False)
        n = len(df_d)
        n_long = max(0, min(int(math.floor(n * top_pct)), max_n))
        n_short = 0
        if mode == "ls":
            n_short = max(0, min(int(math.floor(n * bottom_pct)), max_n))

        # 最少持仓保障
        if n_long == 0 and n >= min_n:
            n_long = min(min_n, n)
        if mode == "ls" and n_short == 0 and n >= min_n:
            n_short = min(min_n, n - n_long)

        long_ret = 0.0
        short_ret = 0.0
        if n_long > 0:
            long_leg = df_d.head(n_long)
            long_ret = float(long_leg["ret"].mean())
        if mode == "ls" and n_short > 0:
            short_leg = df_d.tail(n_short)
            short_ret = float(short_leg["ret"].mean())

        # 成本（双边合计）
        cost_long = (slippage_bps + fee_bps) * 1e-4
        cost_short = (slippage_bps + fee_bps) * 1e-4

        if mode == "long":
            total = long_ret - cost_long
        else:
            total = long_w * (long_ret - cost_long) - short_w * (short_ret + cost_short)

        records.append({
            "date": d,
            "ret_long": long_ret,
            "ret_short": short_ret,
            "ret_total": total,
            "n_long": n_long,
            "n_short": n_short
        })

    if not records:
        return pd.DataFrame(columns=["date", "ret_total"]).set_index("date")

    df_ret = pd.DataFrame(records).set_index("date").sort_index()
    df_ret["nav"] = (1.0 + df_ret["ret_total"]).cumprod()
    return df_ret


def save_nav_plot(nav_df: pd.DataFrame, out_png: Path, title: str):
    ensure_dir(out_png)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(nav_df.index, nav_df["nav"], label="Strategy NAV")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("NAV")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def save_nav_plot_marked(nav_df: pd.DataFrame, window_spans: List[Tuple[pd.Timestamp, pd.Timestamp]], out_png: Path, title: str):
    """
    以半透明背景标注各窗口预测片段
    window_spans: [(start_date, end_date_inclusive), ...]
    """
    ensure_dir(out_png)
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.plot(nav_df.index, nav_df["nav"], color="tab:blue", label="Strategy NAV")
    # 半透明背景带
    colors = ["#FFEDA0", "#AEDFF7", "#C7F2C8", "#FBC4AB", "#D9D7F1", "#D7F2BA"]
    for i, (s, e) in enumerate(window_spans):
        c = colors[i % len(colors)]
        ax.axvspan(s, e, color=c, alpha=0.25, label=(f"win {i+1}: {s.date()} ~ {e.date()}"))
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("NAV")
    ax.grid(True, alpha=0.3)
    # 避免过多 legend 项挤满：限制最多显示 8 个，或合并标签
    if len(window_spans) <= 8:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main():
    cfg = BT_ROLL_CFG
    out_dir = cfg.backtest_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 读取 H5 / 上下文 / 交易日历
    print("[BTRoll] 读取 H5 / 上下文 / 交易日历")
    with h5py.File(cfg.feat_file, "r") as h5:
        d_in = infer_factor_dim(h5)
        date2grp = date_to_group(h5)
        h5_dates = pd.DatetimeIndex(sorted(date2grp.keys()))

    ctx_df = pd.read_parquet(cfg.ctx_file)

    cal = load_calendar(cfg.trading_day_file)
    fridays_all = weekly_fridays(cal)
    mask_bt = (fridays_all >= pd.Timestamp(cfg.bt_start_date)) & (fridays_all <= pd.Timestamp(cfg.bt_end_date))
    fridays_bt = fridays_all[mask_bt]
    if len(fridays_bt) < 1:
        raise RuntimeError("回测周五数量不足。请调整回测区间。")

    # 2) 加载行业映射与模型结构
    ind_map = load_industry_map(cfg.industry_map_file)
    n_ind_known = max(ind_map.values()) + 1
    pad_ind_id = n_ind_known
    ctx_dim = ctx_df.shape[1] if ctx_df.shape[0] > 0 else int(CFG.ctx_dim)

    model_template = GCFNet(
        d_in=d_in, n_ind=n_ind_known, ctx_dim=ctx_dim,
        hidden=CFG.hidden, ind_emb_dim=CFG.ind_emb,
        graph_type=CFG.graph_type, tr_layers=CFG.tr_layers, gat_layers=getattr(CFG, "gat_layers", 1)
    ).to(cfg.device)
    model_template.eval()  # 仅作为结构模板，每窗口加载不同权重

    # 3) 读取日行情并计算下一周收益（close）以及筛选所需表
    price_day_file = cfg.price_day_file
    if not Path(price_day_file).exists():
        raise FileNotFoundError(f"缺少日行情：{price_day_file}，用于计算下周收益/筛选。")
    price_df = pd.read_parquet(price_day_file)
    if not isinstance(price_df.index, pd.MultiIndex):
        price_df = price_df.set_index(["order_book_id", "date"]).sort_index()
    close_pivot = price_df["close"].unstack(0).sort_index()
    avail = close_pivot.index

    # 严格口径：仅当能找到下一周收益时记录 d0（用于收益与筛选）
    weekly_ret, friday_start_map = compute_weekly_returns(close_pivot, fridays_all)

    # 宽松口径：对所有周五都取 d0_any=不晚于 f 的最后一个交易日（即使无法计算收益）
    friday_start_any: Dict[pd.Timestamp, pd.Timestamp] = {}
    for f in fridays_all:
        i0 = avail[avail <= f]
        if len(i0) > 0:
            friday_start_any[f] = i0[-1]

    # 构建筛选闭包（与训练一致）
    filter_fn = None
    stock_info_df = None
    susp_df = None
    st_df = None
    if getattr(cfg, "enable_filters", False):
        # 覆写 CFG 的过滤参数，使 make_filter_fn 读到回测用的门槛
        CFG.enable_filters = bool(cfg.enable_filters)
        CFG.ipo_cut_days = int(cfg.ipo_cut_days)
        CFG.suspended_exclude = bool(cfg.suspended_exclude)
        CFG.st_exclude = bool(cfg.st_exclude)
        CFG.min_daily_turnover = float(cfg.min_daily_turnover)
        CFG.allow_missing_info = bool(cfg.allow_missing_info)
        # 板块开关（与训练一致）
        CFG.include_star_market = bool(getattr(cfg, "include_star_market", True))
        CFG.include_chinext = bool(getattr(cfg, "include_chinext", True))
        CFG.include_bse = bool(getattr(cfg, "include_bse", True))
        CFG.include_neeq = bool(getattr(cfg, "include_neeq", True))

        stock_info_df = load_stock_info(cfg.stock_info_file)
        susp_df = load_flag_table(cfg.is_suspended_file)
        st_df = load_flag_table(cfg.is_st_file)
        filter_fn = make_filter_fn(price_df, stock_info_df, susp_df, st_df)

    # 4) 确定所有可用的窗口锚点 pred_dates（来自训练保存的 best 命名：model_best_YYYYMMDD.pth）
    # 仅使用和回测区间相交的窗口（且后续有至少1个预测周五）
    model_dir = Path(cfg.model_dir)
    step_weeks = int(CFG.step_weeks)

    # 回测有效周五（用于截断预测片段到区间内）
    fridays_bt_set = set(fridays_bt.tolist())

    # 列出所有 best 文件对应的 pred_date
    best_files = sorted(model_dir.glob("model_best_*.pth"))
    window_pred_dates = []
    for f in best_files:
        try:
            tag = f.stem.split("_")[-1]  # YYYYMMDD
            dt = pd.to_datetime(tag)
            window_pred_dates.append((dt, f))
        except Exception:
            continue
    window_pred_dates.sort(key=lambda x: x[0])

    if not window_pred_dates:
        raise RuntimeError(f"未在 {model_dir} 找到任何 model_best_*.pth，无法执行滚动回测。")

    # 5) 逐窗口：对 pred_date 后续 step_weeks 的周五进行预测，片段拼接（区间裁剪，尾部不足尽量使用）
    preds_filtered_all = []
    window_spans = []      # 用于标注图：[(start_dt, end_dt_inclusive)]
    window_metrics = []    # 窗口级指标（对能计算收益的周）

    # 推断 H5 组日期映射
    with h5py.File(cfg.feat_file, "r") as h5:
        d2g = date_to_group(h5)
        h5_dates = pd.DatetimeIndex(sorted(d2g.keys()))

    def map_to_h5_group_date(target_date: pd.Timestamp, h5_idx: pd.DatetimeIndex) -> Optional[pd.Timestamp]:
        """将目标周五映射到不晚于它的 H5 组日期；找不到则返回 None"""
        ok = h5_idx[h5_idx <= target_date]
        return ok[-1] if len(ok) > 0 else None

    # 统一的指标计算（与 backtest.py 一致）
    def calc_stats(nav_df: pd.DataFrame, freq_per_year: int = 52) -> Dict[str, float]:
        if nav_df.empty:
            return {}
        nav = nav_df["nav"].values
        rets = nav_df["ret_total"].values
        total_return = float(nav[-1] - 1.0)
        ann_return = float((1.0 + rets).prod() ** (freq_per_year / max(1, len(rets))) - 1.0)
        vol = float(np.std(rets, ddof=1)) * math.sqrt(freq_per_year) if len(rets) > 1 else 0.0
        sharpe = float(ann_return / vol) if vol > 1e-12 else float("nan")
        peak = -np.inf
        max_dd = 0.0
        for v in nav:
            peak = max(peak, v) if np.isfinite(peak) else v
            dd = (v / peak - 1.0) if peak > 0 else 0.0
            max_dd = min(max_dd, dd)
        max_drawdown = float(-max_dd)
        calmar = float(ann_return / max_drawdown) if max_drawdown > 1e-12 else float("nan")
        return {
            "total_return": total_return,
            "annual_return": ann_return,
            "annual_vol": vol,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "calmar": calmar,
            "n_periods": int(len(rets))
        }

    # 遍历窗口
    for pred_dt, model_path in tqdm(window_pred_dates, desc="滚动回测 - 窗口", unit="win"):
        # 构造该窗口的预测周五序列：pred_dt 开始的 step_weeks 个周五
        if pred_dt not in fridays_all:
            # 若 pred_dt 不在全周五列表中（理论上不应发生），则映射到不晚于它的最近周五
            ok = fridays_all[fridays_all <= pred_dt]
            if len(ok) == 0:
                continue
            start_f = ok[-1]
        else:
            start_f = pred_dt

        # 原始目标区间（长度 step_weeks），再裁剪到 [bt_start, bt_end]
        idx0 = fridays_all.get_indexer([start_f])[0]
        idx1 = min(idx0 + step_weeks, len(fridays_all))  # 左闭右开
        fr_segment = fridays_all[idx0:idx1]
        # 与回测区间取交集（按你要求“按区间裁剪”）
        fr_segment = fr_segment[(fr_segment >= pd.Timestamp(cfg.bt_start_date)) & (fr_segment <= pd.Timestamp(cfg.bt_end_date))]
        if len(fr_segment) == 0:
            continue

        # 加载该窗口的 best 模型与 scaler
        state = torch.load(model_path, map_location=cfg.device, weights_only=False)
        model = GCFNet(
            d_in=d_in, n_ind=n_ind_known, ctx_dim=ctx_dim,
            hidden=CFG.hidden, ind_emb_dim=CFG.ind_emb,
            graph_type=CFG.graph_type, tr_layers=CFG.tr_layers, gat_layers=getattr(CFG, "gat_layers", 1)
        ).to(cfg.device)
        if isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"], strict=False)
            sc_mean = state.get("scaler_mean", None)
            sc_std  = state.get("scaler_std", None)
        else:
            model.load_state_dict(state, strict=False)
            sc_mean = None
            sc_std  = None
        scaler_payload = ScalerPayload(sc_mean, sc_std)
        model.eval()

        # 片段的实际绘图跨度（用于 nav_marked）：从片段内第一个周五对应的净值起点到最后一个“可计算收益”的周五
        # 但标注我们用整个 fr_segment（即使尾周无法计算收益），更直观
        span_start = fr_segment.min()
        span_end   = fr_segment.max()

        # 执行推理
        with h5py.File(cfg.feat_file, "r") as h5:
            for d in fr_segment:
                # 选择用于筛选的口径日：优先严格 d0（能算收益），否则用宽松 d0_any
                start_dt = friday_start_map.get(d, friday_start_any.get(d, None))
                if start_dt is None:
                    continue  # 连本周口径日都没有（行情覆盖不到），跳过

                # 找到该周对应的 H5 组日期：优先同日，其次不晚于 d 的最近组
                g_date = d if d in d2g else map_to_h5_group_date(d, h5_dates)
                if g_date is None:
                    continue
                gk = d2g[g_date]
                g = h5[gk]
                stocks_all = g["stocks"][:].astype(str)
                if len(stocks_all) == 0:
                    continue

                # 快速路径筛选：尽量批量判断（停牌/ST/成交额/板块），剩余用逐只过滤兜底
                pool = stocks_all.tolist()

                if filter_fn is not None and len(pool) > 0:
                    # 1) 板块开关
                    if (not CFG.include_star_market) or (not CFG.include_chinext) or (not CFG.include_bse) or (not CFG.include_neeq):
                        codes = np.asarray(pool, dtype=str)
                        keep_mask = np.ones(codes.shape[0], dtype=bool)

                        if not CFG.include_star_market:
                            is_star = np.char.startswith(codes, "688") | np.char.startswith(codes, "689")
                            keep_mask &= ~is_star

                        if not CFG.include_chinext:
                            is_cx = np.char.startswith(codes, "300") | np.char.startswith(codes, "301")
                            keep_mask &= ~is_cx

                        if not CFG.include_bse:
                            has_xbei = np.char.find(codes, ".XBEI") >= 0
                            has_xbse = np.char.find(codes, ".XBSE") >= 0
                            keep_mask &= ~(has_xbei | has_xbse)

                        if not CFG.include_neeq:
                            suffixes = [".XNE", ".XNEE", ".XNEQ", ".XNEX"]
                            is_neeq = np.zeros_like(keep_mask)
                            for suf in suffixes:
                                is_neeq |= (np.char.find(codes, suf) >= 0)
                            keep_mask &= ~is_neeq

                        pool = codes[keep_mask].tolist()
                        if not pool:
                            continue

                    # 2) 停牌/ST
                    if CFG.suspended_exclude and susp_df is not None and start_dt in susp_df.index:
                        srow = susp_df.loc[start_dt]
                        pool = [s for s in pool if (s in srow.index and int(srow.get(s, 0)) == 0)]
                        if not pool:
                            continue
                    if CFG.st_exclude and st_df is not None and start_dt in st_df.index:
                        srow = st_df.loc[start_dt]
                        pool = [s for s in pool if (s in srow.index and int(srow.get(s, 0)) == 0)]
                        if not pool:
                            continue

                    # 3) 成交额阈值
                    if CFG.min_daily_turnover is not None and CFG.min_daily_turnover > 0:
                        try:
                            df_turn = price_df.xs(start_dt, level=1, drop_level=False)
                            sub = df_turn.loc[(pool, start_dt)] if isinstance(df_turn.index, pd.MultiIndex) else df_turn
                            to_col = None
                            for col in ["total_turnover", "turnover", "amount"]:
                                if col in sub.columns:
                                    to_col = col
                                    break
                            if to_col is not None:
                                ok_codes = sub[sub[to_col] >= CFG.min_daily_turnover].index.get_level_values(0).unique().astype(str).tolist()
                                pool = [s for s in pool if s in ok_codes]
                                if not pool:
                                    continue
                        except Exception:
                            pass

                    # 4) IPO/缺信息等逐只兜底
                    if (not CFG.allow_missing_info) or (CFG.ipo_cut_days and CFG.ipo_cut_days > 0):
                        pool_fast = []
                        for s in pool:
                            try:
                                if filter_fn(start_dt, s):
                                    pool_fast.append(s)
                            except Exception:
                                continue
                        pool = pool_fast
                        if not pool:
                            continue

                if len(pool) == 0:
                    continue  # 本周池内无股票

                # 读取并标准化因子
                X_all = np.asarray(g["factor"][:], dtype=np.float32)  # [N,T,C]
                X_all = ScalerPayload(sc_mean, sc_std).transform(X_all) if ('sc_mean' in locals()) else scaler_payload.transform(X_all)

                # 仅对池内股票打分
                pos = {s: i for i, s in enumerate(stocks_all)}
                idx = [pos[s] for s in pool if s in pos]
                if len(idx) == 0:
                    continue

                X = X_all[idx]
                pool_arr = np.array(pool, dtype=str)

                # 行业ID
                ind = np.asarray([ind_map.get(s, pad_ind_id) for s in pool_arr], dtype=np.int64)

                # 上下文
                if d in ctx_df.index:
                    ctx_vec = ctx_df.loc[d].values.astype(np.float32)
                else:
                    ctx_vec = np.zeros(ctx_dim, dtype=np.float32)
                ctx = np.broadcast_to(ctx_vec, (X.shape[0], ctx_vec.shape[0])).copy()

                # 推理（大批量）
                bs = 8192
                preds = []
                with torch.no_grad():
                    xb = torch.from_numpy(X).to(cfg.device)
                    ib = torch.from_numpy(ind).to(cfg.device)
                    cb = torch.from_numpy(ctx).to(cfg.device)
                    for st in range(0, xb.shape[0], bs):
                        pe = model(xb[st:st+bs], ib[st:st+bs], cb[st:st+bs])
                        preds.append(pe.detach().float().cpu().numpy())
                score = np.concatenate(preds, 0)

                df = pd.DataFrame({"date": d, "stock": pool_arr, "score": score})
                preds_filtered_all.append(df)

        # 记录窗口跨度（用于标注图）
        window_spans.append((span_start, span_end))

    if not preds_filtered_all:
        raise RuntimeError("滚动回测区间内无任何‘池内打分’结果，请检查筛选参数与数据覆盖。")

    pred_df_filtered = pd.concat(preds_filtered_all, ignore_index=True)
    pred_df_filtered = pred_df_filtered.sort_values(["date", "score"], ascending=[True, False])

    # 6) 保存池内预测结果（包含无法计算收益的尾周）
    out_pred_filtered = out_dir / f"predictions_filtered_{cfg.run_name_out}.parquet"
    ensure_dir(out_pred_filtered)
    pred_df_filtered.to_parquet(out_pred_filtered)
    print(f"[BTRoll] 已保存池内预测：{out_pred_filtered}")

    # 7) 构建组合与净值（仅使用能计算收益的周）
    # 限定 weekly_ret 到回测区间内（bt_start_date~bt_end_date）
    weekly_ret_bt = {d: s for d, s in weekly_ret.items()
                     if (d >= pd.Timestamp(cfg.bt_start_date)) and (d <= pd.Timestamp(cfg.bt_end_date))}
    ret_df = build_portfolio(
        pred_df_filtered, weekly_ret_bt,
        mode=cfg.mode,
        top_pct=cfg.top_pct,
        bottom_pct=cfg.bottom_pct,
        min_n=cfg.min_n_stocks,
        max_n=cfg.max_n_stocks,
        long_w=cfg.long_weight,
        short_w=cfg.short_weight,
        slippage_bps=cfg.slippage_bps,
        fee_bps=cfg.fee_bps
    )
    if ret_df.empty:
        print("[BTRoll][警告] 未能生成回测净值（可能尾部仅保存了打分但缺少下一周收益）。")
    else:
        # 7.1 输出净值与图表
        out_nav_csv = out_dir / f"nav_{cfg.run_name_out}.csv"
        ensure_dir(out_nav_csv)
        ret_df[["ret_long", "ret_short", "ret_total", "nav", "n_long", "n_short"]].to_csv(out_nav_csv, float_format="%.8f")
        print(f"[BTRoll] 已保存净值序列：{out_nav_csv}")

        out_png = out_dir / f"nav_{cfg.run_name_out}.png"
        title = f"Rolling Backtest [{cfg.mode}] (step={CFG.step_weeks})"
        save_nav_plot(ret_df, out_png, title=title)
        print(f"[BTRoll] 已保存净值图：{out_png}")

        # 7.2 额外输出标注各窗口片段的 NAV 图
        # 将窗口跨度裁剪到回测区间内以便标注
        spans_clipped = []
        for s, e in window_spans:
            ss = max(pd.Timestamp(cfg.bt_start_date), s)
            ee = min(pd.Timestamp(cfg.bt_end_date), e)
            if ss <= ee:
                spans_clipped.append((ss, ee))
        out_png_marked = out_dir / f"nav_marked_{cfg.run_name_out}.png"
        save_nav_plot_marked(ret_df, spans_clipped, out_png_marked, title=title + " [windows marked]")
        print(f"[BTRoll] 已保存标注窗口片段的净值图：{out_png_marked}")

        # 7.3 指标（总）
        metrics = calc_stats(ret_df)
        out_json = out_dir / f"metrics_{cfg.run_name_out}.json"
        ensure_dir(out_json)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print("[BTRoll] 指标汇总（总）：")
        for k, v in metrics.items():
            print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
        print(f"[BTRoll] 已保存指标JSON：{out_json}")

        # 7.4 窗口级指标（对各窗口片段内能计算收益的周）
        # 方法：对每个片段内的日期子集，取 pred_df_filtered 的记录并与 weekly_ret_bt 计算组合净值，再统计指标
        # 注意：片段内若无可计算收益的周，则该窗口指标 n_periods=0，其余 NaN
        rows_win = []
        for i, (s, e) in enumerate(spans_clipped, 1):
            # 片段内的周五集合（取 pred_df_filtered 的 date 去重）
            dates_seg = sorted(set([d for d in pred_df_filtered["date"].unique() if (d >= s and d <= e)]))
            if len(dates_seg) == 0:
                rows_win.append({"win_id": i, "start": s, "end": e, "n_periods": 0})
                continue
            df_seg = pred_df_filtered[pred_df_filtered["date"].isin(dates_seg)].copy()
            # 构造子组合
            weekly_ret_seg = {d: weekly_ret_bt[d] for d in dates_seg if d in weekly_ret_bt}
            df_nav_seg = build_portfolio(
                df_seg, weekly_ret_seg,
                mode=cfg.mode, top_pct=cfg.top_pct, bottom_pct=cfg.bottom_pct,
                min_n=cfg.min_n_stocks, max_n=cfg.max_n_stocks,
                long_w=cfg.long_weight, short_w=cfg.short_weight,
                slippage_bps=cfg.slippage_bps, fee_bps=cfg.fee_bps
            )
            if df_nav_seg.empty:
                rows_win.append({"win_id": i, "start": s, "end": e, "n_periods": 0})
            else:
                m = calc_stats(df_nav_seg)
                m_row = {"win_id": i, "start": s, "end": e}
                m_row.update(m)
                rows_win.append(m_row)

        df_win = pd.DataFrame(rows_win)
        out_win_csv = out_dir / f"metrics_by_window_{cfg.run_name_out}.csv"
        ensure_dir(out_win_csv)
        df_win.to_csv(out_win_csv, index=False)
        print(f"[BTRoll] 已保存窗口级指标：{out_win_csv}")


if __name__ == "__main__":
    main()