# coding: utf-8
"""
btr_gen_scores.py
滚动回测（推理阶段拆分版）：逐窗口加载 best 模型，对每个目标周五进行筛选与打分，保存池内打分 parquet
输出：
- backtest_rolling/{run_name}/predictions_filtered_{run_name_out}.parquet
备注：
- 逻辑大量复用 backtest_rolling.py 的前半段（数据加载、筛选、推理）；去掉组合与指标相关内容
"""

import math, json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import h5py
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")

from backtest_rolling_config import BT_ROLL_CFG
from config import CFG
from model import GCFNet
from utils import load_calendar, weekly_fridays, load_industry_map
from train_utils import make_filter_fn, load_stock_info, load_flag_table

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
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

def date_to_group(h5: h5py.File) -> Dict[pd.Timestamp, str]:
    df = list_h5_groups(h5)
    return {row["date"]: row["group"] for _, row in df.iterrows()}

def infer_factor_dim(h5: h5py.File) -> int:
    if "factor_cols" in h5.attrs:
        return int(len(h5.attrs["factor_cols"]))
    for k in h5.keys():
        if "factor" in h5[k]:
            arr = np.asarray(h5[k]["factor"][:])
            arr = np.squeeze(arr)
            return int(arr.shape[-1])
    raise RuntimeError("无法推断因子维度")

class ScalerPayload:
    def __init__(self, mean: Optional[np.ndarray], std: Optional[np.ndarray]):
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32)
        self.std  = None if std  is None else np.asarray(std,  dtype=np.float32)
        if self.mean is not None and self.std is not None:
            if self.mean.ndim == 1:
                self.mean = self.mean.reshape(1,1,-1)
            elif self.mean.ndim == 2:
                self.mean = self.mean.reshape(1, self.mean.shape[0], self.mean.shape[1]).mean(axis=1, keepdims=False).reshape(1,1,-1)
            elif self.mean.ndim != 3:
                raise ValueError(f"scaler_mean 维度不支持: {self.mean.shape}")
            if self.std.ndim == 1:
                self.std = self.std.reshape(1,1,-1)
            elif self.std.ndim == 2:
                self.std = self.std.reshape(1, self.std.shape[0], self.std.shape[1]).mean(axis=1, keepdims=False).reshape(1,1,-1)
            elif self.std.ndim != 3:
                raise ValueError(f"scaler_std 维度不支持: {self.std.shape}")

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.squeeze(np.asarray(X))
        if self.mean is None or self.std is None:
            return X.astype(np.float32)
        return ((X - self.mean) / (self.std + 1e-6)).astype(np.float32)

def main():
    cfg = BT_ROLL_CFG
    out_dir = cfg.backtest_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[BTR-GenScores] 读取 H5 / 上下文 / 交易日历")
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

    ind_map = load_industry_map(cfg.industry_map_file)
    n_ind_known = max(ind_map.values()) + 1
    pad_ind_id = n_ind_known
    ctx_dim = ctx_df.shape[1] if ctx_df.shape[0] > 0 else int(CFG.ctx_dim)

    model_template = GCFNet(
        d_in=d_in, n_ind=n_ind_known, ctx_dim=ctx_dim,
        hidden=CFG.hidden, ind_emb_dim=CFG.ind_emb,
        graph_type=CFG.graph_type, tr_layers=CFG.tr_layers, gat_layers=getattr(CFG, "gat_layers", 1)
    ).to(cfg.device)
    model_template.eval()

    # 加载原始日线（用于过滤成交额/停牌等）
    price_day_file = cfg.price_day_file
    if not Path(price_day_file).exists():
        raise FileNotFoundError(f"缺少日行情：{price_day_file}")
    price_df = pd.read_parquet(price_day_file)
    if not isinstance(price_df.index, pd.MultiIndex):
        price_df = price_df.set_index(["order_book_id", "date"]).sort_index()
    close_pivot = price_df["close"].unstack(0).sort_index()
    avail = close_pivot.index

    # 确定筛选口径日（严格能算收益与宽松仅口径日）
    friday_start_any: Dict[pd.Timestamp, pd.Timestamp] = {}
    for f in fridays_all:
        i0 = avail[avail <= f]
        if len(i0) > 0:
            friday_start_any[f] = i0[-1]

    filter_fn = None
    stock_info_df = None
    susp_df = None
    st_df = None
    if getattr(cfg, "enable_filters", False):
        CFG.enable_filters = bool(cfg.enable_filters)
        CFG.ipo_cut_days = int(cfg.ipo_cut_days)
        CFG.suspended_exclude = bool(cfg.suspended_exclude)
        CFG.st_exclude = bool(cfg.st_exclude)
        CFG.min_daily_turnover = float(cfg.min_daily_turnover)
        CFG.allow_missing_info = bool(cfg.allow_missing_info)
        CFG.include_star_market = bool(getattr(cfg, "include_star_market", True))
        CFG.include_chinext = bool(getattr(cfg, "include_chinext", True))
        CFG.include_bse = bool(getattr(cfg, "include_bse", True))
        CFG.include_neeq = bool(getattr(cfg, "include_neeq", True))
        stock_info_df = load_stock_info(cfg.stock_info_file)
        susp_df = load_flag_table(cfg.is_suspended_file)
        st_df = load_flag_table(cfg.is_st_file)
        filter_fn = make_filter_fn(price_df, stock_info_df, susp_df, st_df)

    model_dir = Path(cfg.model_dir)
    step_weeks = int(CFG.step_weeks)

    best_files = sorted(model_dir.glob("model_best_*.pth"))
    window_pred_dates = []
    for f in best_files:
        try:
            tag = f.stem.split("_")[-1]
            dt = pd.to_datetime(tag)
            window_pred_dates.append((dt, f))
        except Exception:
            continue
    window_pred_dates.sort(key=lambda x: x[0])
    if not window_pred_dates:
        raise RuntimeError(f"未在 {model_dir} 找到任何 model_best_*.pth。")

    preds_filtered_all = []
    window_spans = []

    with h5py.File(cfg.feat_file, "r") as h5:
        d2g = date_to_group(h5)
        h5_dates = pd.DatetimeIndex(sorted(d2g.keys()))

    def map_to_h5_group_date(target_date: pd.Timestamp, h5_idx: pd.DatetimeIndex) -> Optional[pd.Timestamp]:
        ok = h5_idx[h5_idx <= target_date]
        return ok[-1] if len(ok) > 0 else None

    for pred_dt, model_path in tqdm(window_pred_dates, desc="生成打分 - 窗口", unit="win"):
        if pred_dt not in fridays_all:
            ok = fridays_all[fridays_all <= pred_dt]
            if len(ok) == 0:
                continue
            start_f = ok[-1]
        else:
            start_f = pred_dt

        idx0 = fridays_all.get_indexer([start_f])[0]
        idx1 = min(idx0 + step_weeks, len(fridays_all))
        fr_segment = fridays_all[idx0:idx1]
        fr_segment = fr_segment[(fr_segment >= pd.Timestamp(cfg.bt_start_date)) & (fr_segment <= pd.Timestamp(cfg.bt_end_date))]
        if len(fr_segment) == 0:
            continue

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
            sc_mean = None; sc_std = None
        scaler_payload = ScalerPayload(sc_mean, sc_std)
        model.eval()

        span_start = fr_segment.min()
        span_end   = fr_segment.max()

        with h5py.File(cfg.feat_file, "r") as h5:
            for d in fr_segment:
                start_dt = friday_start_any.get(d, None)
                if start_dt is None:
                    continue
                g_date = d if d in d2g else map_to_h5_group_date(d, h5_dates)
                if g_date is None:
                    continue
                gk = d2g[g_date]
                g = h5[gk]
                stocks_all = g["stocks"][:].astype(str)
                if len(stocks_all) == 0:
                    continue

                pool = stocks_all.tolist()

                if filter_fn is not None and len(pool) > 0:
                    # 板块/停牌/ST/成交额/IPO兜底（同 backtest_rolling）
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

                    if CFG.suspended_exclude and (start_dt in (load_flag_table(cfg.is_suspended_file) or pd.DataFrame()).index if False else True):
                        # 为避免重复读取，直接使用外部已读的 susp_df/st_df
                        pass
                    if CFG.suspended_exclude and (susp_df is not None) and (start_dt in susp_df.index):
                        srow = susp_df.loc[start_dt]
                        pool = [s for s in pool if (s in srow.index and int(srow.get(s, 0)) == 0)]
                        if not pool:
                            continue
                    if CFG.st_exclude and (st_df is not None) and (start_dt in st_df.index):
                        srow = st_df.loc[start_dt]
                        pool = [s for s in pool if (s in srow.index and int(srow.get(s, 0)) == 0)]
                        if not pool:
                            continue
                    if CFG.min_daily_turnover is not None and CFG.min_daily_turnover > 0:
                        try:
                            df_turn = price_df.xs(start_dt, level=1, drop_level=False)
                            sub = df_turn.loc[(pool, start_dt)] if isinstance(df_turn.index, pd.MultiIndex) else df_turn
                            to_col = None
                            for col in ["total_turnover", "turnover", "amount"]:
                                if col in sub.columns:
                                    to_col = col; break
                            if to_col is not None:
                                ok_codes = sub[sub[to_col] >= CFG.min_daily_turnover].index.get_level_values(0).unique().astype(str).tolist()
                                pool = [s for s in pool if s in ok_codes]
                                if not pool:
                                    continue
                        except Exception:
                            pass
                    if (not CFG.allow_missing_info) or (CFG.ipo_cut_days and CFG.ipo_cut_days > 0):
                        pool_fast = []
                        for s in pool:
                            try:
                                if make_filter_fn(price_df, stock_info_df, susp_df, st_df)(start_dt, s):
                                    pool_fast.append(s)
                            except Exception:
                                continue
                        pool = pool_fast
                        if not pool:
                            continue

                if len(pool) == 0:
                    continue

                X_all = np.asarray(g["factor"][:], dtype=np.float32)
                X_all = scaler_payload.transform(X_all)

                pos = {s: i for i, s in enumerate(stocks_all)}
                idx = [pos[s] for s in pool if s in pos]
                if len(idx) == 0:
                    continue

                X = X_all[idx]
                pool_arr = np.array(pool, dtype=str)

                ind = np.asarray([ind_map.get(s, pad_ind_id) for s in pool_arr], dtype=np.int64)

                if d in ctx_df.index:
                    ctx_vec = ctx_df.loc[d].values.astype(np.float32)
                else:
                    ctx_vec = np.zeros(ctx_dim, dtype=np.float32)
                ctx = np.broadcast_to(ctx_vec, (X.shape[0], ctx_vec.shape[0])).copy()

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

        window_spans.append((span_start, span_end))

    if not preds_filtered_all:
        raise RuntimeError("区间内无任何池内打分结果。")

    pred_df_filtered = pd.concat(preds_filtered_all, ignore_index=True)
    pred_df_filtered = pred_df_filtered.sort_values(["date", "score"], ascending=[True, False])

    out_pred_filtered = out_dir / f"predictions_filtered_{cfg.run_name_out}.parquet"
    ensure_dir(out_pred_filtered)
    pred_df_filtered.to_parquet(out_pred_filtered)
    print(f"[BTR-GenScores] 已保存池内打分：{out_pred_filtered}")

if __name__ == "__main__":
    main()