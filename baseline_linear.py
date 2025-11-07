# coding: utf-8
"""
baseline_linear_fast.py
在 baseline_linear.py 基础上针对速度做了系统优化：
- 组内特征构造完全矢量化
- 标准化 with_mean=False，避免一次大拷贝；或可直接关闭标准化
- 缓存每个 group 的 (tickers, X, y)，训练/预测复用
- 自适应并行线程数（物理核）
- 可选序列截断 CFG.seq_trunc（默认 60）
"""

import os, h5py, numpy as np, pandas as pd, psutil
from pathlib import Path
from typing import Dict, Tuple, Optional

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm, trange

from config import CFG
from backtest_rolling_config import BT_ROLL_CFG
from utils import load_calendar, weekly_fridays
from train_utils import make_filter_fn, load_stock_info, load_flag_table

# ----------------- 并行/数值环境 -----------------
def _setup_threads():
    # 优先使用物理核的 80%（至少 1，至多 16），避免与系统/IO 抢占
    phys = psutil.cpu_count(logical=False) or 4
    n = max(1, min(int(phys * 0.8), 16))
    env_keys = ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]
    for k in env_keys:
        if k not in os.environ:
            os.environ[k] = str(n)
    return n

_NUM_THREADS = int(getattr(CFG, "num_threads", _setup_threads()))

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

TRAIN_YEARS = int(CFG.train_years)    # 如 5
STEP_WEEKS  = int(CFG.step_weeks)     # 如 52

# 可选：时间序列截断长度和模型正则
SEQ_TRUNC = int(getattr(CFG, "seq_trunc", 60))
RIDGE_ALPHA = float(getattr(CFG, "linear_alpha", 1.0))
USE_SCALER = bool(getattr(CFG, "linear_use_scaler", True))  # 是否使用标准化
SCALER_WITH_STD = bool(getattr(CFG, "linear_scaler_with_std", True))  # 只做缩放不做中心化

# ----------------- 工具 -----------------
def list_h5_groups(h5: h5py.File) -> pd.DataFrame:
    rows = []
    for k in h5.keys():
        if not k.startswith("date_"): continue
        d = h5[k].attrs.get("date", None)
        if d is None: continue
        if isinstance(d, bytes): d = d.decode("utf-8")
        rows.append({"group": k, "date": pd.to_datetime(d)})
    if not rows: return pd.DataFrame(columns=["group","date"])
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

def date_to_group(h5: h5py.File) -> Dict[pd.Timestamp, str]:
    df = list_h5_groups(h5)
    return {row["date"]: row["group"] for _, row in df.iterrows()}

def map_to_h5_group_date(target_date: pd.Timestamp, h5_idx: pd.DatetimeIndex) -> Optional[pd.Timestamp]:
    ok = h5_idx[h5_idx <= target_date]
    return ok[-1] if len(ok) > 0 else None

def build_feature_matrix_for_group_fast(
    g, d: pd.Timestamp, label_df: pd.DataFrame, filter_fn
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    矢量化构造特征:
    输入:
      - g: h5 group, 含 datasets: stocks [N], factor [N, T, C] 或 [N, C]
      - d: 日期
    输出:
      - X: [M, F] float32
      - y: [M] float32
      - tickers: [M] str
    """
    stocks = g["stocks"][:].astype(str)
    X_all = np.asarray(g["factor"][:])  # [N, T, C] or [N, C]
    X_all = np.squeeze(X_all)

    # 过滤可交易与有标签的股票（矢量化）
    # label_df 索引为 (date, stock)
    if isinstance(label_df.index, pd.MultiIndex):
        try:
            sub = label_df.xs(d, level=0, drop_level=False)
        except KeyError:
            return np.zeros((0,1), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.array([], dtype=str)
        valid_set = set(sub.index.get_level_values(1).astype(str).values.tolist())
    else:
        sub = label_df[label_df["date"] == d]
        valid_set = set(sub["stock"].astype(str).values.tolist())

    # 初筛：是否在标签集合
    in_label = np.fromiter((s in valid_set for s in stocks), dtype=bool, count=len(stocks))

    # 二次筛：可选的交易过滤函数（若提供）
    if filter_fn is not None:
        tradable = np.fromiter((filter_fn(d, s) for s in stocks), dtype=bool, count=len(stocks))
        keep = in_label & tradable
    else:
        keep = in_label

    if not keep.any():
        return np.zeros((0,1), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.array([], dtype=str)

    X_kept = X_all[keep]
    kept_stocks = stocks[keep]

    # 将 [N, T, C] 统一到 [N, T, C]
    if X_kept.ndim == 2:
        X_kept = X_kept[:, None, :]  # [N, 1, C]
    # 序列截断到最近 SEQ_TRUNC
    if SEQ_TRUNC and X_kept.shape[1] > SEQ_TRUNC:
        X_kept = X_kept[:, -SEQ_TRUNC:, :]

    # 矢量化构造摘要特征
    # X_kept: [M, T, C]
    vT = X_kept[:, -1, :]                          # [M, C]
    mean = np.nanmean(X_kept, axis=1)              # [M, C]
    std  = np.nanstd (X_kept, axis=1)              # [M, C]

    def delta(k):
        if X_kept.shape[1] > k:
            return vT - X_kept[:, -1-k, :]
        return np.zeros_like(vT)

    d1  = delta(1)
    d5  = delta(5)
    d10 = delta(10)

    def recent_mean(k):
        kk = min(X_kept.shape[1], k)
        return np.nanmean(X_kept[:, -kk:, :], axis=1)

    m5  = recent_mean(5)
    m10 = recent_mean(10)
    m20 = recent_mean(20)

    X_feat = np.concatenate([vT, mean, std, d1, d5, d10, m5, m10, m20], axis=1)
    X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # y 按 kept_stocks 顺序取
    # sub 可能很大，直接从 sub.loc[(d, s)] 逐个取会慢；预先构建映射更快
    if isinstance(sub.index, pd.MultiIndex):
        sub_now = sub.droplevel(0)  # index=stock
        y = sub_now.reindex(kept_stocks)["next_week_return"].to_numpy(dtype=np.float32, na_value=np.nan)
    else:
        sub_now = sub.set_index("stock")
        y = sub_now.reindex(kept_stocks)["next_week_return"].to_numpy(dtype=np.float32, na_value=np.nan)

    mask = ~np.isnan(y)
    if not mask.any():
        return np.zeros((0,1), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.array([], dtype=str)

    return X_feat[mask], y[mask], kept_stocks[mask]

# ----------------- 主流程 -----------------
def main():
    # 日历
    cal = load_calendar(TRADING_DAY_FILE)
    fridays_all = weekly_fridays(cal)
    mask_bt = (fridays_all >= START) & (fridays_all <= END)
    fridays_bt = fridays_all[mask_bt]
    if len(fridays_bt) == 0:
        raise RuntimeError("无回测周五")

    # 标签
    label_df = pd.read_parquet(LABEL_PATH)
    if not isinstance(label_df.index, pd.MultiIndex):
        label_df = label_df.set_index(["date","stock"]).sort_index()

    # 交易过滤依赖
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

        # group 级缓存：避免重复构造
        # 缓存内容：{gk: (date, tickers[np.ndarray], X[np.ndarray], y[np.ndarray])}
        cache = {}

        preds_all = []

        train_weeks = TRAIN_YEARS * 52
        step_weeks  = STEP_WEEKS

        pbar_windows = trange(train_weeks, len(fridays_bt), step_weeks, desc="[Linear] Rolling fast", leave=True)
        for i in pbar_windows:
            train_dates = fridays_bt[i - train_weeks : i]
            pred_dates  = fridays_bt[i : min(i + step_weeks, len(fridays_bt))]

            # 组装训练集（从缓存取或构建）
            X_train_li, y_train_li = [], []
            for d in tqdm(train_dates, desc="  build train set", leave=False):
                g_date = d if d in d2g else map_to_h5_group_date(d, h5_dates)
                if g_date is None: continue
                gk = d2g[g_date]
                if gk in cache:
                    _, _, Xg, yg = cache[gk]
                else:
                    g = h5[gk]
                    Xg, yg, tickers = build_feature_matrix_for_group_fast(g, g_date, label_df, filter_fn)
                    cache[gk] = (g_date, tickers, Xg, yg)
                if Xg.shape[0] == 0: continue
                X_train_li.append(Xg); y_train_li.append(yg)

            if not X_train_li:
                pbar_windows.write(f"[Linear] 窗口{i} 训练样本为空，跳过")
                continue

            X_train = np.concatenate(X_train_li, axis=0)
            y_train = np.concatenate(y_train_li, axis=0)
            pbar_windows.write(f"[Linear] window@{i} train_samples={X_train.shape[0]} dim={X_train.shape[1]}")

            # 标准化（仅缩放，不中心化，减少一次大内存拷贝；线性回归不一定需要，但通常略增稳健性）
            if USE_SCALER:
                scaler = StandardScaler(with_mean=False, with_std=SCALER_WITH_STD, copy=False)
                Xs = scaler.fit_transform(X_train)
            else:
                scaler = None
                Xs = X_train  # 直接用

            # Ridge 自适应线程：sklearn 会遵循 OMP_NUM_THREADS
            model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True, random_state=42)

            # 拟合
            model.fit(Xs, y_train)

            # 预测阶段
            for d in tqdm(pred_dates, desc="  predict", leave=False):
                g_date = d if d in d2g else map_to_h5_group_date(d, h5_dates)
                if g_date is None: continue
                gk = d2g[g_date]
                if gk in cache:
                    _, tickers, Xg, _ = cache[gk]
                else:
                    g = h5[gk]
                    Xg, yg, tickers = build_feature_matrix_for_group_fast(g, g_date, label_df, filter_fn)
                    cache[gk] = (g_date, tickers, Xg, yg)
                if Xg.shape[0] == 0: continue
                Xp = scaler.transform(Xg) if scaler is not None else Xg
                pred = model.predict(Xp).astype(np.float32)

                preds_all.append(pd.DataFrame({
                    "date": d,
                    "stock": tickers.astype(str),
                    "score": pred
                }))

        if not preds_all:
            raise RuntimeError("无任何线性模型预测输出")
        pred_df = pd.concat(preds_all, ignore_index=True)
        pred_df.sort_values(["date","score"], ascending=[True, False], inplace=True)
        out_path = OUT_DIR / f"predictions_filtered_{RUN.run_name_out}.parquet"
        pred_df.to_parquet(out_path)
        print(f"[Linear] 已保存: {out_path}")

if __name__ == "__main__":
    main()