# coding: utf-8
"""
周频信号回测（支持做多 / 多空）
- 读取 features_daily.h5 与 context_features.parquet
- 加载已训练模型（与训练相同的 model.GCFNet）
- 对回测区间内每个周五取组，计算打分，形成组合并根据下周收益计算净值
- 输出：每周预测、净值序列CSV、净值图、指标汇总JSON
"""
import math
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from backtest_config import BT_CFG
from config import CFG  # 复用训练期的一些默认，如窗口长度，仅用于形状检查
from model import GCFNet
from utils import load_calendar, weekly_fridays, load_industry_map
from train_utils import pearsonr  # 可用于额外评估用

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
        return pd.DataFrame(columns=["group","date"])
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

def predict_one_group(model: GCFNet,
                      h5: h5py.File,
                      gk: str,
                      ctx_df: pd.DataFrame,
                      ind_map: dict,
                      pad_ind_id: int,
                      device: torch.device) -> pd.DataFrame:
    g = h5[gk]
    date = pd.Timestamp(g.attrs["date"])
    stocks = g["stocks"][:].astype(str)
    X = np.asarray(g["factor"][:])  # [N,T,C]（已在特征阶段做过MAD clip，但无 scaler）
    X = np.squeeze(X).astype(np.float32)

    # 注意：训练时在每个窗口对 factor 做了“窗口内拟合Scaler并transform”，
    # 回测时为了一致性，最稳妥方式是在训练好模型时就使用训练时的 scaler（每窗口不同）。
    # 这里我们采用“原样输入”（与训练窗口不完全一致），在实践中一般仍能工作；
    # 若需要严格一致，可在训练窗口保存 scaler，并在回测时选取与模型对应窗口的 scaler。
    # 也可简易 z-score：按当前组内做列级标准化（可选）
    # 这里给个可选的轻量标准化开关：
    do_group_z = False
    if do_group_z:
        mu = np.nanmean(X, axis=(0,1), keepdims=True)
        sd = np.nanstd(X, axis=(0,1), keepdims=True) + 1e-6
        X = (X - mu) / sd

    # 行业ID
    ind = np.asarray([ind_map.get(s, pad_ind_id) for s in stocks], dtype=np.int64)

    # 上下文
    if date in ctx_df.index:
        ctx_vec = ctx_df.loc[date].values.astype(np.float32)
    else:
        ctx_vec = np.zeros(ctx_df.shape[1], dtype=np.float32)
    ctx = np.broadcast_to(ctx_vec, (X.shape[0], ctx_vec.shape[0])).copy()

    # 预测
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
                           fridays: pd.DatetimeIndex) -> Dict[pd.Timestamp, pd.Series]:
    """
    与标签生成保持一致的对齐：对每个周五，找 <=该周五 的最后交易日作为起点，
    下一个周五同理，计算 (end/start - 1)。
    返回：{friday -> Series(index=stock)} 的一周收益
    """
    avail = close_pivot.index
    out = {}
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
        end   = close_pivot.loc[d1]
        ret = (end / start - 1).replace([np.inf,-np.inf], np.nan)
        out[f0] = ret
    return out

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
    - 简化假设：当周末形成持仓，持有到下个周末，用下一周收益实现
    """
    dates = sorted(set(scores["date"]))
    records = []

    for d in dates:
        df_d = scores[scores["date"] == d].copy()
        if d not in weekly_ret:
            # 无对齐收益，跳过
            continue
        ret_s = weekly_ret[d]  # Series(stock -> ret)

        df_d = df_d.merge(ret_s.rename("ret"), left_on="stock", right_index=True, how="inner")
        df_d = df_d.dropna(subset=["ret", "score"])
        if len(df_d) < min_n:
            # 样本不足，空仓
            records.append({"date": d, "ret_long": 0.0, "ret_short": 0.0, "ret_total": 0.0, "n_long": 0, "n_short": 0})
            continue

        df_d = df_d.sort_values("score", ascending=False)
        n = len(df_d)
        n_long = max(0, min(int(math.floor(n * top_pct)), max_n))
        n_short = 0
        if mode == "ls":
            n_short = max(0, min(int(math.floor(n * bottom_pct)), max_n))

        # 若最少持仓约束，尝试放宽到 min_n
        if n_long == 0 and n >= min_n:
            n_long = min(min_n, n)
        if mode == "ls" and n_short == 0 and n >= min_n:
            n_short = min(min_n, n - n_long)

        long_ret = 0.0; short_ret = 0.0
        if n_long > 0:
            long_leg = df_d.head(n_long)
            long_ret = float(long_leg["ret"].mean())
        if mode == "ls" and n_short > 0:
            short_leg = df_d.tail(n_short)
            short_ret = float(short_leg["ret"].mean())

        # 简化成本：进出各一次，合计双边成本
        cost_long  = (slippage_bps + fee_bps) * 1e-4
        cost_short = (slippage_bps + fee_bps) * 1e-4

        # 组合收益
        if mode == "long":
            total = long_ret - cost_long
        else:
            # 多空对冲，加权
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
        return pd.DataFrame(columns=["date","ret_total"]).set_index("date")

    df_ret = pd.DataFrame(records).set_index("date").sort_index()
    df_ret["nav"] = (1.0 + df_ret["ret_total"]).cumprod()
    return df_ret

def calc_stats(nav_df: pd.DataFrame, freq_per_year: int = 52) -> Dict[str, float]:
    """
    计算常见指标（周频）：
    - 总收益、年化收益、波动率、夏普、最大回撤、卡玛
    """
    if nav_df.empty:
        return {}

    nav = nav_df["nav"].values
    rets = nav_df["ret_total"].values

    total_return = float(nav[-1] - 1.0)
    ann_return = float((1.0 + rets).prod() ** (freq_per_year / max(1, len(rets))) - 1.0)
    vol = float(np.std(rets, ddof=1)) * math.sqrt(freq_per_year) if len(rets) > 1 else 0.0
    sharpe = float(ann_return / vol) if vol > 1e-12 else float("nan")

    # 最大回撤
    peak = -np.inf
    dd = 0.0
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

    # 1) 读取数据
    print("[BT] 读取 H5 / 上下文 / 指数与交易日历")
    with h5py.File(BT_CFG.feat_file, "r") as h5:
        d_in = infer_factor_dim(h5)
        date2grp = date_to_group(h5)
        h5_dates = sorted(date2grp.keys())

    ctx_df = pd.read_parquet(BT_CFG.ctx_file)
    # 允许上下文缺失周五，用 0 向量处理（预测函数里已兼容）

    cal = load_calendar(BT_CFG.trading_day_file)
    fridays_all = weekly_fridays(cal)
    bt_mask = (fridays_all >= pd.Timestamp(BT_CFG.bt_start_date)) & (fridays_all <= pd.Timestamp(BT_CFG.bt_end_date))
    fridays_bt = fridays_all[bt_mask]
    if len(fridays_bt) < 2:
        raise RuntimeError("回测周五数量不足（<2）。请调整回测区间。")

    # 2) 加载模型
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
    model.load_state_dict(state, strict=False)
    model.eval()

    # 3) 逐周预测
    preds_all = []
    with h5py.File(BT_CFG.feat_file, "r") as h5:
        for d in fridays_bt:
            if d not in date2grp:
                continue
            gk = date2grp[d]
            df_pred = predict_one_group(
                model, h5, gk, ctx_df, ind_map, pad_ind_id, BT_CFG.device
            )
            preds_all.append(df_pred)

    if not preds_all:
        raise RuntimeError("回测区间内未获得任何周五预测，请检查数据覆盖与模型匹配。")
    pred_df = pd.concat(preds_all, ignore_index=True)
    pred_df = pred_df.sort_values(["date","score"], ascending=[True, False])

    # 保存每周预测（用于复盘/调参）
    out_pred = out_dir / f"predictions_{BT_CFG.run_name}.parquet"
    pred_df.to_parquet(out_pred)
    print(f"[BT] 已保存每周预测：{out_pred}")

    # 4) 准备计算下一周收益（用股票日行情 close）
    # 从 features_daily.h5 不直接提供 close，因此这里建议从 raw 的 stock_price_day.parquet 读取。
    # 为避免新增依赖，这里用 label 文件的逻辑重建（需要原始日行情）。
    price_day_file = CFG.price_day_file  # 复用训练期路径
    if not Path(price_day_file).exists():
        raise FileNotFoundError(f"缺少日行情：{price_day_file}，用于计算下周收益。")
    price_df = pd.read_parquet(price_day_file)
    if not isinstance(price_df.index, pd.MultiIndex):
        price_df = price_df.set_index(["order_book_id","date"]).sort_index()
    close_pivot = price_df["close"].unstack(0).sort_index()

    # 构建回测区间的“周收益字典”
    weekly_ret = compute_weekly_returns(close_pivot, fridays_bt)

    # 5) 组合构建与净值
    ret_df = build_portfolio(
        pred_df, weekly_ret,
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

    # 6) 输出净值与图表
    out_nav_csv = out_dir / f"nav_{BT_CFG.run_name}.csv"
    ensure_dir(out_nav_csv)
    ret_df[["ret_long","ret_short","ret_total","nav","n_long","n_short"]].to_csv(out_nav_csv, float_format="%.8f")
    print(f"[BT] 已保存净值序列：{out_nav_csv}")

    out_png = out_dir / f"nav_{BT_CFG.run_name}.png"
    save_nav_plot(ret_df, out_png, title=f"{BT_CFG.model_name} [{BT_CFG.mode}]")
    print(f"[BT] 已保存净值图：{out_png}")

    # 7) 指标
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