# coding: utf-8
"""
quick_infer_intraday.py
在任意交易日 D（包括非周五）做“推理-only”：
- 生成 D 日全市场打分（降序）
- 生成 D 日目标权重快照（today_target）
说明：
- 不改训练/标签/周五采样，只做前向推理
- 口径与回测一致（因子、MAD 裁剪、标准化、上下文、过滤、中性化、选股/赋权）
"""

import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import h5py
from tqdm import tqdm

# ======== 顶部可修改变量 ========
ASOF_DATE = "2025-11-06"          # 例: "2025-11-06"；None 表示使用“价格数据”中的最新可用交易日
MODEL_TAG = None          # 例: "20251107"；None 表示自动选“最近一个不晚于 D 的 model_best_YYYYMMDD.pth”
FORCE_NO_FILTERS = False  # True 跳过停牌/ST/IPO/成交额等过滤；False 使用回测同样过滤
OVERRIDE_WEIGHT_MODE = None  # "equal" 或 "score"；None 则使用 BT_ROLL_CFG.weight_mode
# ==============================

# 项目内依赖
from config import CFG
from backtest_rolling_config import BT_ROLL_CFG as CFG_BT
from model import GCFNet
from utils import load_industry_map, mad_clip
from train_utils import make_filter_fn, HAS_CUDA, load_stock_info, load_flag_table
from optimize_position import _neutralize_week_scores, _select_and_weight
from feature_engineering import calc_factors_one_stock_full

# -------- 工具函数 --------
def _ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def _pick_best_model_for_date(model_dir: Path, d: pd.Timestamp, tag_override: Optional[str]) -> Path:
    if tag_override:
        p = model_dir / f"model_best_{tag_override}.pth"
        if not p.exists():
            raise FileNotFoundError(f"未找到指定模型：{p}")
        return p
    cands = sorted(model_dir.glob("model_best_*.pth"))
    if not cands:
        raise FileNotFoundError(f"目录无 best 模型：{model_dir}")
    ok: List[Tuple[pd.Timestamp, Path]] = []
    for f in cands:
        try:
            tag = f.stem.split("_")[-1]
            dt = pd.to_datetime(tag)
            if dt <= d:
                ok.append((dt, f))
        except Exception:
            continue
    if not ok:
        # 若均晚于 D，则退回最早一个（也可改成报错）
        return cands[0]
    ok.sort(key=lambda x: x[0])
    return ok[-1][1]

def _read_factor_cols_from_h5(h5_path: Path) -> List[str]:
    if not h5_path.exists():
        raise FileNotFoundError(f"缺少特征仓以获取列名：{h5_path}")
    with h5py.File(h5_path, "r") as h5f:
        if "factor_cols" not in h5f.attrs:
            raise RuntimeError("features_daily.h5 缺少 factor_cols 属性")
        arr = np.array(h5f.attrs["factor_cols"])
        cols = [s.decode("utf-8") if isinstance(s, (bytes, bytearray)) else str(s) for s in arr]
    return cols

def _broadcast_ctx_for_date(D: pd.Timestamp) -> pd.Series:
    # 生成 D 的上下文向量：从 context_features.parquet 取“<=D 的最后一行”
    ctx_path = CFG.processed_dir / "context_features.parquet"
    if not ctx_path.exists():
        raise FileNotFoundError(f"缺少上下文文件：{ctx_path}，请先运行 feature_context.py 更新到 D")
    ctx_df = pd.read_parquet(ctx_path).sort_index()
    if not isinstance(ctx_df.index, pd.DatetimeIndex):
        raise RuntimeError("context_features 索引必须是 DatetimeIndex")
    idx = ctx_df.index[ctx_df.index <= D]
    if len(idx) == 0:
        raise RuntimeError(f"context_features 中不含 {D.date()} 及之前的记录")
    return ctx_df.loc[idx[-1]]

def _load_daily_price() -> pd.DataFrame:
    df = pd.read_parquet(CFG.price_day_file)
    # 统一 MultiIndex(order_book_id, date)
    if not isinstance(df.index, pd.MultiIndex):
        idx_cols = [c for c in df.columns if c in ["order_book_id", "date", "datetime"]]
        if set(idx_cols) >= {"order_book_id"} and (("date" in idx_cols) or ("datetime" in idx_cols)):
            if "datetime" in df.columns and "date" not in df.columns:
                df = df.copy()
                df["date"] = pd.to_datetime(df["datetime"])
            elif "date" in df.columns:
                df = df.copy()
                df["date"] = pd.to_datetime(df["date"])
            df = df.set_index(["order_book_id", "date"]).sort_index()
        else:
            raise RuntimeError("price_day_file 既非 MultiIndex 也无 [order_book_id, date/datetime] 列")
    # 统一第二级索引名
    names = list(df.index.names)
    if names[1] != "date":
        try:
            df.index = df.index.set_names(["order_book_id", "date"])
        except Exception:
            pass
    # 尽量确保日期为 DatetimeIndex
    try:
        df = df.copy()
        df.index = pd.MultiIndex.from_arrays(
            [df.index.get_level_values(0), pd.to_datetime(df.index.get_level_values(1))],
            names=["order_book_id", "date"]
        )
    except Exception:
        pass
    return df.sort_index()

def _compute_factors_for_window(df_all: pd.DataFrame,
                                D: pd.Timestamp,
                                factor_cols: List[str],
                                max_lookback: int,
                                daily_window: int,
                                stocks: List[str]) -> Dict[str, np.ndarray]:
    # 截取局部片段：多取一些冗余（2*max_lookback）
    start = D - pd.Timedelta(days=max_lookback*2)
    df_slice = df_all[(df_all.index.get_level_values("date") >= start) &
                      (df_all.index.get_level_values("date") <= D)]
    out: Dict[str, np.ndarray] = {}
    print(f"[INFO] 计算因子窗口: 股票数={len(stocks)}, 窗口长度={daily_window}, 片段={start.date()}~{D.date()}")
    for s in tqdm(stocks, desc="factors", ncols=80):
        if s not in df_slice.index.get_level_values(0):
            continue
        try:
            hist = df_slice.loc[s].sort_index()
        except KeyError:
            continue
        fct = calc_factors_one_stock_full(hist)
        if fct.empty:
            continue
        # 取尾部窗口
        fct = fct.reindex(columns=factor_cols)
        fct = fct.replace([np.inf, -np.inf], np.nan)
        fct = fct.dropna(how="all")
        fct_win = fct[fct.index <= D].tail(daily_window)
        if len(fct_win) != daily_window:
            continue
        vals = fct_win.values.astype(np.float32)
        if not np.isfinite(vals).any():
            continue
        out[s] = vals
    return out

def _load_filters_inputs() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    # 尝试加载三张表：stock_info、停牌、ST
    stock_info_df = None
    susp_df = None
    st_df = None
    try:
        stock_info_df = load_stock_info(Path(CFG.stock_info_file))
        if stock_info_df is None or stock_info_df.empty:
            print("[WARN] stock_info 加载为空，将放宽 allow_missing_info")
            stock_info_df = None
    except Exception as e:
        print(f"[WARN] 读取 stock_info 异常：{e}，将放宽 allow_missing_info")
        stock_info_df = None

    try:
        susp_df = load_flag_table(Path(CFG.is_suspended_file))
        if susp_df is not None and not isinstance(susp_df.index, pd.DatetimeIndex):
            susp_df.index = pd.to_datetime(susp_df.index)
    except Exception as e:
        print(f"[WARN] 读取 is_suspended 异常：{e}")
        susp_df = None

    try:
        st_df = load_flag_table(Path(CFG.is_st_file))
        if st_df is not None and not isinstance(st_df.index, pd.DatetimeIndex):
            st_df.index = pd.to_datetime(st_df.index)
    except Exception as e:
        print(f"[WARN] 读取 is_st_stock 异常：{e}")
        st_df = None

    # 若无法加载 stock_info，避免全量剔除：放宽 allow_missing_info
    if stock_info_df is None:
        CFG.allow_missing_info = True
    return stock_info_df, susp_df, st_df

def main():
    # 1) 读取必要元信息（以价格数据对齐 D）
    factor_cols = _read_factor_cols_from_h5(CFG.feat_file)
    daily_df = _load_daily_price()

    # 可用日期全集（来自价格数据）
    all_dates = pd.DatetimeIndex(sorted(daily_df.index.get_level_values("date").unique()))
    if len(all_dates) == 0:
        raise RuntimeError("price_day_file 中没有任何日期数据")

    # 初始 D：用 ASOF_DATE 或 价格数据的最大日期
    if ASOF_DATE is None:
        D = all_dates.max()
        print(f"[INFO] 未指定 ASOF_DATE，使用价格数据最新日期 D={D.date()}")
    else:
        D_req = pd.to_datetime(ASOF_DATE)
        if D_req not in all_dates:
            pos = all_dates.searchsorted(D_req, side="right") - 1
            if pos < 0:
                raise RuntimeError(f"ASOF_DATE={D_req.date()} 早于价格数据最早日期 {all_dates.min().date()}，请先更新行情")
            D = all_dates[pos]
            print(f"[WARN] 指定 ASOF_DATE={D_req.date()} 不在价格数据中，改用最近可用 D={D.date()}")
        else:
            D = D_req
            print(f"[INFO] 使用指定 ASOF_DATE D={D.date()}")

    # 2) 选择模型（不晚于 D 的最近 best；或用手动指定）
    best_path = _pick_best_model_for_date(CFG.model_dir, D, MODEL_TAG)
    print(f"[INFO] 推理日期 D = {D.date()}, 使用模型 = {best_path.name}")

    # 3) 取当日股票集合
    try:
        day_slice = daily_df.xs(D, level="date", drop_level=False)
    except KeyError:
        mask = (daily_df.index.get_level_values("date") == D)
        day_slice = daily_df[mask]
    stocks_all = sorted(day_slice.index.get_level_values(0).unique())
    print(f"[INFO] 原始股票数（当日有行情）= {len(stocks_all)}")

    # 4) 样本过滤输入数据
    stock_info_df, susp_df, st_df = (None, None, None)
    if not FORCE_NO_FILTERS:
        stock_info_df, susp_df, st_df = _load_filters_inputs()

    # 5) 构造过滤闭包
    if FORCE_NO_FILTERS:
        filt_fn = None
        print("[INFO] 跳过过滤（FORCE_NO_FILTERS=True）")
    else:
        filt_fn = make_filter_fn(daily_df.reset_index(), stock_info_df, susp_df, st_df)
        print(f"[INFO] 过滤配置: allow_missing_info={CFG.allow_missing_info}, "
              f"min_daily_turnover={getattr(CFG,'min_daily_turnover',0)}, "
              f"boards: STAR={CFG.include_star_market}, CHINEXT={CFG.include_chinext}, "
              f"BSE={CFG.include_bse}, NEEQ={CFG.include_neeq}")

    # 6) 执行过滤
    stocks = []
    for s in stocks_all:
        ok = True
        if filt_fn is not None:
            try:
                ok = bool(filt_fn(D, s))
            except Exception:
                ok = True
        if ok:
            stocks.append(s)
    print(f"[INFO] 过滤后股票数 = {len(stocks)}")
    if not stocks:
        raise RuntimeError("过滤后无可用股票。若异常，请检查 stock_info / is_suspended / is_st 数据或临时设置 FORCE_NO_FILTERS=True 以排查。")

    # 7) 计算因子窗口 X[D, stk]
    feats_map = _compute_factors_for_window(daily_df, D, factor_cols, CFG.max_lookback, CFG.daily_window, stocks)
    if len(feats_map) == 0:
        raise RuntimeError("无法为任何股票生成窗口特征，请检查行情覆盖与窗口长度。")
    stocks_ok = sorted(feats_map.keys())
    X = np.stack([feats_map[s] for s in stocks_ok], axis=0)  # [N,T,C]
    X = mad_clip(X).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"[INFO] 有效窗口股票数 = {len(stocks_ok)}")

    # 8) 加载模型与标准化
    device = CFG.device
    ind_map = load_industry_map(CFG.industry_map_file)
    n_ind_known = (max(ind_map.values()) + 1) if len(ind_map) else 0
    pad_ind_id = n_ind_known

    ctx_row = _broadcast_ctx_for_date(D).astype(np.float32)
    ctx_dim = ctx_row.shape[0]
    ctx = np.broadcast_to(ctx_row.values, (len(stocks_ok), ctx_dim)).astype(np.float32)

    model = GCFNet(
        d_in=len(factor_cols), n_ind=n_ind_known, ctx_dim=ctx_dim,
        hidden=CFG.hidden, ind_emb_dim=CFG.ind_emb,
        graph_type=CFG.graph_type, tr_layers=CFG.tr_layers, gat_layers=getattr(CFG, "gat_layers", 1)
    ).to(device)

    payload = torch.load(best_path, map_location=device, weights_only=False)
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    if isinstance(payload, dict) and ("scaler_mean" in payload) and ("scaler_std" in payload):
        mean = np.asarray(payload["scaler_mean"], dtype=np.float32)
        std  = np.asarray(payload["scaler_std"],  dtype=np.float32)
        print(f"[INFO] 使用 best 权重内随附的 Scaler")
    else:
        mean = np.nanmean(X, axis=(0,1), keepdims=True).astype(np.float32)
        std  = (np.nanstd (X, axis=(0,1), keepdims=True) + 1e-6).astype(np.float32)
        print(f"[WARN] best 权重未附带 Scaler，使用当前批次估计（存在口径差异）")

    Xn = (X - mean) / std
    Xn = np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0)
    ind_id = np.asarray([ind_map.get(s, pad_ind_id) for s in stocks_ok], dtype=np.int64)

    # 9) 前向推理（分批，tqdm）
    bs = max(1024, min(8192, len(stocks_ok)))
    n_steps = int(math.ceil(len(stocks_ok) / bs))
    preds: List[np.ndarray] = []
    print(f"[INFO] 前向推理: N={len(stocks_ok)}, batch_size={bs}, steps={n_steps}")
    with torch.no_grad():
        for st in tqdm(range(0, len(stocks_ok), bs), desc="infer", ncols=80):
            xb = torch.from_numpy(Xn[st:st+bs]).to(device, non_blocking=HAS_CUDA)
            ib = torch.from_numpy(ind_id[st:st+bs]).to(device, non_blocking=HAS_CUDA)
            cb = torch.from_numpy(ctx[st:st+bs]).to(device, non_blocking=HAS_CUDA)
            p = model(xb, ib, cb).detach().float().cpu().numpy()
            preds.append(p)
    score = np.concatenate(preds, axis=0).astype(np.float32)

    # 10) 行业中性化与选股/赋权（与回测口径一致）
    df_scores = pd.DataFrame({
        "date": D,
        "stock": stocks_ok,
        "score": score
    })
    do_neutral = bool(getattr(CFG_BT, "neutralize_enable", False))
    neutral_method = str(getattr(CFG_BT, "neutralize_method", "ols_resid")).lower()
    add_intercept  = bool(getattr(CFG_BT, "neutralize_add_intercept", True))
    clip_pct       = float(getattr(CFG_BT, "neutralize_clip_pct", 0.0))

    if do_neutral:
        dfn = _neutralize_week_scores(df_scores, ind_map, add_intercept, clip_pct, method=neutral_method)
        col_use = "score_neutral" if dfn["score_neutral"].notna().any() else "score"
    else:
        dfn = df_scores.copy()
        dfn["score_neutral"] = dfn["score"]
        col_use = "score"

    weight_mode = (OVERRIDE_WEIGHT_MODE or getattr(CFG_BT, "weight_mode", "score")).lower()
    dsel = _select_and_weight(
        df_week=dfn.assign(score=dfn["score"].astype(float)),
        weight_mode=weight_mode,
        top_pct=float(CFG_BT.top_pct),
        max_n=int(CFG_BT.max_n_stocks),
        min_n=int(CFG_BT.min_n_stocks),
        filter_negative_scores_long=bool(CFG_BT.filter_negative_scores_long),
        col_use=col_use
    )
    today_target = dsel.copy()
    today_target["date"] = D
    today_target["signal_date"] = D
    today_target = today_target[["date","stock","weight","score","score_neutral","signal_date"]]
    today_target = today_target.sort_values(["weight","stock"], ascending=[False, True]).reset_index(drop=True)
    print(f"[INFO] today_target 股票数 = {len(today_target)}")

    # 11) 导出
    out_dir = Path(CFG_BT.backtest_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = D.strftime("%Y%m%d")

    # 全市场打分（降序）
    df_out_scores = dfn[["date","stock","score","score_neutral"]].copy()
    df_out_scores = df_out_scores.sort_values(["score"], ascending=[False]).reset_index(drop=True)
    f_scores = out_dir / f"scores_intraday_{CFG_BT.run_name_out}_{tag}.csv"
    _ensure_dir(f_scores)
    df_out_scores.to_csv(f_scores, index=False, float_format="%.8f")

    # 今日目标权重
    f_today = out_dir / f"today_target_{CFG_BT.run_name_out}_{weight_mode}_{tag}.csv"
    _ensure_dir(f_today)
    today_target.to_csv(f_today, index=False, float_format="%.8f")

    print(f"[DONE] D={D.date()} | scores -> {f_scores.name} | today_target -> {f_today.name}")

if __name__ == "__main__":
    main()