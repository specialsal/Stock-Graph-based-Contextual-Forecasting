# coding: utf-8
"""
周频信号回测（支持做多 / 多空）— 含筛选版与原始版预测输出 + tqdm 进度显示
- 读取 features_daily.h5 与 context_features.parquet
- 加载已训练模型（与训练相同的 model.GCFNet）
- 对回测区间内每个周五取组，计算打分
- 在组合构建前，按与训练一致的口径进行样本过滤（去 ST、停牌、成交额过低、IPO未满等，参数来自 backtest_config）
- 保存两份预测：原始未筛选 predictions_raw_*.parquet 与 筛选后 predictions_filtered_*.parquet
- 根据下周收益计算净值，输出：净值CSV、净值图、指标JSON

新增（问题B修复）：
- 从模型checkpoint读取 scaler_mean / scaler_std，并在回测推理时对 [N,T,C] 做与训练一致的标准化。
- 若checkpoint中不存在 scaler 参数，则打印警告并退化为不做标准化（保持兼容）。
"""
import math
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

from backtest_config import BT_CFG
from config import CFG  # 复用训练期的一些默认（隐层维度、图设置等）
from model import GCFNet
from utils import load_calendar, weekly_fridays, load_industry_map
from train_utils import pearsonr
# 训练期的筛选工具
from train_utils import make_filter_fn, load_stock_info, load_flag_table

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


def predict_one_group(model: GCFNet,
                      h5: h5py.File,
                      gk: str,
                      ctx_df: pd.DataFrame,
                      ind_map: dict,
                      pad_ind_id: int,
                      device: torch.device,
                      scaler_payload: ScalerPayload) -> pd.DataFrame:
    g = h5[gk]
    date = pd.Timestamp(g.attrs["date"])
    stocks = g["stocks"][:].astype(str)
    X = np.asarray(g["factor"][:])  # [N,T,C]（特征阶段已MAD clip）
    X = scaler_payload.transform(X)  # === 修复点：与训练一致的标准化 ===

    # 行业ID
    ind = np.asarray([ind_map.get(s, pad_ind_id) for s in stocks], dtype=np.int64)

    # 上下文
    if date in ctx_df.index:
        ctx_vec = ctx_df.loc[date].values.astype(np.float32)
    else:
        ctx_vec = np.zeros(ctx_df.shape[1], dtype=np.float32)
    ctx = np.broadcast_to(ctx_vec, (X.shape[0], ctx_vec.shape[0])).copy()

    # 推理
    with torch.no_grad():
        xb = torch.from_numpy(X).to(device)
        ib = torch.from_numpy(ind).to(device)
        cb = torch.from_numpy(ctx).to(device)
        preds = []
        bs = 4096
        for st in range(0, xb.shape[0], bs):
            pe = model(xb[st:st+bs], ib[st:st+bs], cb[st:st+bs])
            preds.append(pe.detach().float().cpu().numpy())
        score = np.concatenate(preds, 0)

    df = pd.DataFrame({
        "date": date,
        "stock": stocks,
        "score": score
    })
    return df


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


def main():
    out_dir = BT_CFG.processed_dir / "backtest"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 读取 H5 / 上下文 / 交易日历
    print("[BT] 读取 H5 / 上下文 / 交易日历")
    with h5py.File(BT_CFG.feat_file, "r") as h5:
        d_in = infer_factor_dim(h5)
        date2grp = date_to_group(h5)

    ctx_df = pd.read_parquet(BT_CFG.ctx_file)

    cal = load_calendar(BT_CFG.trading_day_file)
    fridays_all = weekly_fridays(cal)
    bt_mask = (fridays_all >= pd.Timestamp(BT_CFG.bt_start_date)) & (fridays_all <= pd.Timestamp(BT_CFG.bt_end_date))
    fridays_bt = fridays_all[bt_mask]
    if len(fridays_bt) < 2:
        raise RuntimeError("回测周五数量不足（<2）。请调整回测区间。")

    # 2) 加载模型 + 读取训练期的 scaler（若有）
    model_path = (BT_CFG.model_dir / BT_CFG.model_name)
    if not model_path.exists():
        raise FileNotFoundError(f"模型不存在：{model_path}")
    print(f"[BT] 加载模型: {model_path.name}")

    # 行业映射
    ind_map = load_industry_map(BT_CFG.industry_map_file)
    n_ind_known = max(ind_map.values()) + 1
    pad_ind_id = n_ind_known

    # ctx 维度
    ctx_dim = ctx_df.shape[1] if ctx_df.shape[0] > 0 else int(CFG.ctx_dim)

    model = GCFNet(
        d_in=d_in, n_ind=n_ind_known, ctx_dim=ctx_dim,
        hidden=CFG.hidden, ind_emb_dim=CFG.ind_emb,
        graph_type=CFG.graph_type, tr_layers=CFG.tr_layers, gat_layers=getattr(CFG, "gat_layers", 1)
    ).to(BT_CFG.device)

    state = torch.load(model_path, map_location=BT_CFG.device)
    # 兼容两类保存方式：直接state_dict或payload字典
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"], strict=False)
        sc_mean = state.get("scaler_mean", None)
        sc_std  = state.get("scaler_std", None)
    else:
        model.load_state_dict(state, strict=False)
        sc_mean = None
        sc_std  = None
    scaler_payload = ScalerPayload(sc_mean, sc_std)
    if scaler_payload.mean is None or scaler_payload.std is None:
        print("[BT][提示] 未在模型文件中发现 scaler_mean/std，推理将不做标准化（与原实现一致）。"
              "为与训练完全一致，建议在训练保存时一起保存 scaler 参数。")
    model.eval()

    # 3) 逐周预测（tqdm）
    preds_all = []
    with h5py.File(BT_CFG.feat_file, "r") as h5:
        for d in tqdm(fridays_bt, desc="预测周五打分", unit="w"):
            if d not in date2grp:
                continue
            gk = date2grp[d]
            df_pred = predict_one_group(
                model, h5, gk, ctx_df, ind_map, pad_ind_id, BT_CFG.device, scaler_payload
            )
            preds_all.append(df_pred)

    if not preds_all:
        raise RuntimeError("回测区间内未获得任何周五预测，请检查数据覆盖与模型匹配。")
    pred_df_raw = pd.concat(preds_all, ignore_index=True)
    pred_df_raw = pred_df_raw.sort_values(["date", "score"], ascending=[True, False])

    # 4) 读取日行情并计算下一周收益（close）
    price_day_file = BT_CFG.price_day_file
    if not Path(price_day_file).exists():
        raise FileNotFoundError(f"缺少日行情：{price_day_file}，用于计算下周收益/筛选。")
    price_df = pd.read_parquet(price_day_file)
    if not isinstance(price_df.index, pd.MultiIndex):
        price_df = price_df.set_index(["order_book_id", "date"]).sort_index()
    close_pivot = price_df["close"].unstack(0).sort_index()

    # 周收益与起点交易日映射
    weekly_ret, friday_start_map = compute_weekly_returns(close_pivot, fridays_bt)

    # 5) 构建筛选闭包（与训练一致），并应用于每周的预测集合（以起点交易日口径）
    pred_df_filtered = pred_df_raw.copy()
    if getattr(BT_CFG, "enable_filters", False):
        # 覆写 CFG 的过滤参数，使 make_filter_fn 读到回测用的门槛
        CFG.enable_filters = bool(BT_CFG.enable_filters)
        CFG.ipo_cut_days = int(BT_CFG.ipo_cut_days)
        CFG.suspended_exclude = bool(BT_CFG.suspended_exclude)
        CFG.st_exclude = bool(BT_CFG.st_exclude)
        CFG.min_daily_turnover = float(BT_CFG.min_daily_turnover)
        CFG.allow_missing_info = bool(BT_CFG.allow_missing_info)

        stock_info_df = load_stock_info(BT_CFG.stock_info_file)
        susp_df = load_flag_table(BT_CFG.is_suspended_file)
        st_df = load_flag_table(BT_CFG.is_st_file)
        filter_fn = make_filter_fn(price_df, stock_info_df, susp_df, st_df)

        if filter_fn is not None:
            keep_rows = []
            for d, grp in tqdm(pred_df_filtered.groupby("date"), desc="按口径日筛选", unit="w"):
                if d not in friday_start_map:
                    # 找不到对齐起点，直接跳过该周（保留为空，本周不持仓）
                    continue
                start_dt = friday_start_map[d]  # 该周的起点交易日
                mask = []
                stocks = grp["stock"].tolist()
                for s in stocks:
                    try:
                        ok = filter_fn(start_dt, s)
                    except Exception:
                        ok = False
                    mask.append(bool(ok))
                sub = grp.loc[mask]
                if len(sub) > 0:
                    keep_rows.append(sub)
            pred_df_filtered = pd.concat(keep_rows, ignore_index=True) if keep_rows else pred_df_filtered.iloc[0:0]
            pred_df_filtered = pred_df_filtered.sort_values(["date", "score"], ascending=[True, False])

    # 6) 保存两份预测结果
    out_pred_raw = out_dir / f"predictions_raw_{BT_CFG.run_name}.parquet"
    out_pred_filtered = out_dir / f"predictions_filtered_{BT_CFG.run_name}.parquet"
    ensure_dir(out_pred_raw)
    ensure_dir(out_pred_filtered)
    pred_df_raw.to_parquet(out_pred_raw)
    pred_df_filtered.to_parquet(out_pred_filtered)
    print(f"[BT] 已保存原始预测：{out_pred_raw}")
    print(f"[BT] 已保存筛选后预测：{out_pred_filtered}")

    # 7) 用筛选后的预测构建组合与净值
    ret_df = build_portfolio(
        pred_df_filtered, weekly_ret,
        mode=BT_CFG.mode,
        top_pct=BT_CFG.top_pct,
        bottom_pct=BT_CFG.bottom_pct,
        min_n=BT_CFG.min_n_stocks,
        max_n=BT_CFG.max_n_stocks,
        long_w=BT_CFG.long_weight,
        short_w=BT_CFG.short_weight,
        slippage_bps=BT_CFG.slippage_bps,
        fee_bps=BT_CFG.fee_bps
    )
    if ret_df.empty:
        raise RuntimeError("未能生成回测净值，请检查回测参数与数据。")

    # 8) 输出净值与图表
    out_nav_csv = out_dir / f"nav_{BT_CFG.run_name}.csv"
    ensure_dir(out_nav_csv)
    ret_df[["ret_long", "ret_short", "ret_total", "nav", "n_long", "n_short"]].to_csv(out_nav_csv, float_format="%.8f")
    print(f"[BT] 已保存净值序列：{out_nav_csv}")

    out_png = out_dir / f"nav_{BT_CFG.run_name}.png"
    save_nav_plot(ret_df, out_png, title=f"{BT_CFG.model_name} [{BT_CFG.mode}]")
    print(f"[BT] 已保存净值图：{out_png}")

    # 9) 指标
    metrics = calc_stats(ret_df)
    out_json = out_dir / f"metrics_{BT_CFG.run_name}.json"
    ensure_dir(out_json)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("[BT] 指标汇总：")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"[BT] 已保存指标JSON：{out_json}")


if __name__ == "__main__":
    main()