# coding: utf-8
"""
baseline_lgbm_fast.py with tqdm and speedups (merged fixes)
- 修复 MultiIndex 筛选 KeyError
- 特征构造前清洗/裁剪，避免溢出告警
- LightGBM 早停使用 callbacks，兼容旧版本
- 预测时兼容 best_iteration/current_iteration
"""
import os, h5py, numpy as np, pandas as pd, psutil
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import lightgbm as lgb
from tqdm import tqdm, trange

from config import CFG
from backtest_rolling_config import BT_ROLL_CFG
from utils import load_calendar, weekly_fridays
from train_utils import make_filter_fn, load_stock_info, load_flag_table

# ----------------- 配置 -----------------
RUN = BT_ROLL_CFG
OUT_DIR = RUN.backtest_dir
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEAT_H5 = RUN.feat_file
LABEL_PATH = RUN.label_file
TRADING_DAY_FILE = RUN.trading_day_file
PRICE_DAY_FILE = RUN.price_day_file

START = pd.Timestamp(CFG.start_date)
END   = pd.Timestamp(RUN.bt_end_date)

TRAIN_YEARS = int(CFG.train_years)
STEP_WEEKS  = int(CFG.step_weeks)
SEQ_TRUNC   = int(getattr(CFG, "seq_trunc", 60))  # 构造摘要特征时最多用最近 T（可按需）

NUM_THREADS = max(1, min(int(psutil.cpu_count(logical=False) * 0.8) or 8, 16))

# 特征数值裁剪阈值（可按数据规模调整）
CLIP_ABS = float(getattr(CFG, "feat_clip_abs", 1e6))

# ----------------- 工具 -----------------
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
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

def date_to_group(h5: h5py.File) -> Dict[pd.Timestamp, str]:
    df = list_h5_groups(h5)
    return {row["date"]: row["group"] for _, row in df.iterrows()}

def map_to_h5_group_date(target_date: pd.Timestamp, h5_idx: pd.DatetimeIndex) -> Optional[pd.Timestamp]:
    ok = h5_idx[h5_idx <= target_date]
    return ok[-1] if len(ok) > 0 else None

def build_feature_matrix_for_group_fast(X_all: np.ndarray) -> np.ndarray:
    """
    将组内原始因子 [N, T, C] 或 [N, C] 构造成摘要特征（矢量化），并进行数值清洗与裁剪，避免溢出。
    返回 float32 的特征矩阵 [N, F]
    """
    # 确保 float32，降低内存与溢出风险
    X = np.asarray(X_all, dtype=np.float32)

    if X.ndim == 2:
        # 二维：先清洗与裁剪后直接返回
        X = np.where(np.isfinite(X), X, 0.0)
        if CLIP_ABS is not None and CLIP_ABS > 0:
            X = np.clip(X, -CLIP_ABS, CLIP_ABS)
        return X

    # 三维：按时间截断
    if X.shape[1] > SEQ_TRUNC:
        X = X[:, -SEQ_TRUNC:, :]

    # 非有限值置零 + 裁剪，避免后续 nanmean/nanstd 溢出警告
    X = np.where(np.isfinite(X), X, 0.0)
    if CLIP_ABS is not None and CLIP_ABS > 0:
        X = np.clip(X, -CLIP_ABS, CLIP_ABS)

    vT = X[:, -1, :]                            # [N, C]
    mean = np.nanmean(X, axis=1)                # [N, C]
    std  = np.nanstd (X, axis=1)                # [N, C]

    def delta(k):
        if X.shape[1] > k:
            return vT - X[:, -1-k, :]
        return np.zeros_like(vT)

    d1  = delta(1)
    d5  = delta(5)
    d10 = delta(10)

    def recent_mean(k):
        kk = min(X.shape[1], k)
        return np.nanmean(X[:, -kk:, :], axis=1)

    m5  = recent_mean(5)
    m10 = recent_mean(10)
    m20 = recent_mean(20)

    feats = np.concatenate([vT, mean, std, d1, d5, d10, m5, m10, m20], axis=1)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    if CLIP_ABS is not None and CLIP_ABS > 0:
        feats = np.clip(feats, -CLIP_ABS, CLIP_ABS)
    return feats.astype(np.float32)

def build_group_xy(h5: h5py.File, gk: str, label_df: pd.DataFrame, filter_fn):
    """
    构造单个 group 的 (X, y, tickers)
    - 修复 MultiIndex 筛选 KeyError：对当日样本使用 xs(d, level=0)，并按 stocks reindex
    - 与过滤函数一起构建 keep_mask，避免逐个循环
    """
    g = h5[gk]
    d = pd.Timestamp(g.attrs["date"])
    stocks = g["stocks"][:].astype(str)
    X_all = np.asarray(g["factor"][:])  # [N, T, C] or [N, C]
    X_all = np.squeeze(X_all)

    # 将 label 可用性与过滤一次性矢量化
    if isinstance(label_df.index, pd.MultiIndex):
        try:
            # 索引变为 stock
            sub = label_df.xs(d, level=0)
        except KeyError:
            return np.zeros((0,1), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.array([], dtype=str)
        # 按组内股票顺序对齐
        sub = sub.reindex(stocks)
        # 确保列存在
        if "next_week_return" not in sub.columns:
            raise KeyError("label_df 缺少列: next_week_return")
        keep_mask = ~sub["next_week_return"].isna().to_numpy()
    else:
        # 非 MultiIndex 兜底
        sub = label_df[label_df["date"] == d].set_index("stock").reindex(stocks)
        if "next_week_return" not in sub.columns:
            raise KeyError("label_df 缺少列: next_week_return")
        keep_mask = ~sub["next_week_return"].isna().to_numpy()

    if filter_fn is not None:
        tradable = np.fromiter((filter_fn(d, s) for s in stocks), dtype=bool, count=len(stocks))
        keep_mask &= tradable

    if not keep_mask.any():
        return np.zeros((0,1), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.array([], dtype=str)

    X_kept = X_all[keep_mask]
    feats = build_feature_matrix_for_group_fast(X_kept)

    kept_stocks = stocks[keep_mask]
    y = sub.loc[keep_mask, "next_week_return"].to_numpy(dtype=np.float32)

    return feats, y, kept_stocks

# ----------------- 主流程 -----------------
def main():
    cal = load_calendar(TRADING_DAY_FILE)
    fridays_all = weekly_fridays(cal)
    mask_bt = (fridays_all >= START) & (fridays_all <= END)
    fridays_bt = fridays_all[mask_bt]
    if len(fridays_bt) == 0:
        raise RuntimeError("无回测周五")

    label_df = pd.read_parquet(LABEL_PATH)
    if not isinstance(label_df.index, pd.MultiIndex):
        label_df = label_df.set_index(["date","stock"]).sort_index()

    price_df = pd.read_parquet(PRICE_DAY_FILE)
    if not isinstance(price_df.index, pd.MultiIndex):
        price_df = price_df.set_index(["order_book_id","date"]).sort_index()

    stock_info_df = load_stock_info(RUN.stock_info_file)
    susp_df = load_flag_table(RUN.is_suspended_file)
    st_df   = load_flag_table(RUN.is_st_file)
    filter_fn = make_filter_fn(price_df, stock_info_df, susp_df, st_df) if RUN.enable_filters else None

    with h5py.File(FEAT_H5, "r") as h5:
        d2g = date_to_group(h5)
        h5_dates = pd.DatetimeIndex(sorted(d2g.keys()))
        cache_xy: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        preds_all = []
        train_weeks = TRAIN_YEARS * 52
        step_weeks  = STEP_WEEKS

        pbar_windows = trange(train_weeks, len(fridays_bt), step_weeks, desc="[LGBM] fast rolling", leave=True)
        for i in pbar_windows:
            train_dates = fridays_bt[i - train_weeks : i]
            pred_dates  = fridays_bt[i : min(i + step_weeks, len(fridays_bt))]

            # 构造训练集（带缓存）
            X_tr_li, y_tr_li = [], []
            for d in tqdm(train_dates, desc="  build train (fridays)", leave=False):
                g_date = d if d in d2g else map_to_h5_group_date(d, h5_dates)
                if g_date is None:
                    continue
                gk = d2g[g_date]
                if gk in cache_xy:
                    Xg, yg, _ = cache_xy[gk]
                else:
                    Xg, yg, tickers = build_group_xy(h5, gk, label_df, filter_fn)
                    cache_xy[gk] = (Xg, yg, tickers)
                if Xg.shape[0] == 0:
                    continue
                X_tr_li.append(Xg); y_tr_li.append(yg)
            if not X_tr_li:
                pbar_windows.write(f"[LGBM] window {i} no train samples, skip")
                continue
            X_train = np.concatenate(X_tr_li, axis=0)
            y_train = np.concatenate(y_tr_li, axis=0)
            pbar_windows.write(f"[LGBM] window@{i} train_samples={X_train.shape[0]} dim={X_train.shape[1]}")

            # 构造验证集（最近几周）用于早停
            if len(train_dates) >= 8:
                valid_dates = train_dates[-8:]
                X_va_li, y_va_li = [], []
                for d in valid_dates:
                    g_date_v = d if d in d2g else map_to_h5_group_date(d, h5_dates)
                    if g_date_v is None:
                        continue
                    gk_v = d2g[g_date_v]
                    if gk_v in cache_xy:
                        Xg, yg, _ = cache_xy[gk_v]
                    else:
                        Xg, yg, tickers = build_group_xy(h5, gk_v, label_df, filter_fn)
                        cache_xy[gk_v] = (Xg, yg, tickers)
                    if Xg.shape[0] == 0:
                        continue
                    X_va_li.append(Xg); y_va_li.append(yg)
                if X_va_li:
                    X_valid = np.concatenate(X_va_li, axis=0); y_valid = np.concatenate(y_va_li, axis=0)
                else:
                    X_valid, y_valid = None, None
            else:
                X_valid, y_valid = None, None

            params = {
                "objective": "regression",
                "metric": "l2",
                "learning_rate": 0.05,
                "num_leaves": 64,
                "max_depth": -1,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 1,
                "min_data_in_leaf": 100,
                "lambda_l2": 1.0,
                "verbose": -1,
                "seed": 42,
                "num_threads": NUM_THREADS,
            }
            lgb_train = lgb.Dataset(X_train, label=y_train, free_raw_data=False)

            callbacks = []
            valid_sets = [lgb_train]
            valid_names = ["train"]
            if X_valid is not None:
                lgb_valid = lgb.Dataset(X_valid, label=y_valid, free_raw_data=False)
                valid_sets = [lgb_train, lgb_valid]
                valid_names = ["train", "valid"]
                # 使用回调早停，兼容不同版本 lightgbm
                callbacks.append(lgb.early_stopping(stopping_rounds=100, verbose=False))
                # 如需控制日志频率，可启用下一行（部分旧版不支持 period=0）
                # callbacks.append(lgb.log_evaluation(period=100))

            model = lgb.train(
                params,
                lgb_train,
                num_boost_round=2000,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=callbacks,
            )

            # 预测
            for d in tqdm(pred_dates, desc="  predict (fridays)", leave=False):
                g_date = d if d in d2g else map_to_h5_group_date(d, h5_dates)
                if g_date is None:
                    continue
                gk = d2g[g_date]
                if gk in cache_xy:
                    Xg, yg, tickers = cache_xy[gk]
                else:
                    Xg, yg, tickers = build_group_xy(h5, gk, label_df, filter_fn)
                    cache_xy[gk] = (Xg, yg, tickers)
                if Xg.shape[0] == 0:
                    continue
                # 兼容不同版本：优先 best_iteration；否则尝试 current_iteration()；再退化为 None
                best_it = getattr(model, "best_iteration", None)
                if best_it is None:
                    best_it = model.current_iteration() if hasattr(model, "current_iteration") else None
                pred = model.predict(Xg, num_iteration=best_it)
                df_out = pd.DataFrame({"date": d, "stock": tickers.astype(str), "score": pred.astype(np.float32)})
                preds_all.append(df_out)

        if not preds_all:
            raise RuntimeError("无任何LGBM预测输出")
        pred_df = pd.concat(preds_all, ignore_index=True).sort_values(["date","score"], ascending=[True, False])
        out_path = OUT_DIR / f"predictions_filtered_{RUN.run_name_out}.parquet"
        pred_df.to_parquet(out_path)
        print(f"[LGBM] 已保存: {out_path}")

if __name__ == "__main__":
    main()