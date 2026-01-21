# coding: utf-8
"""
baseline_lstm_fast.py with tqdm, AMP and speedups (方案A：保留5年训练 + 跨窗口缓存)
- 保留每窗完整5年训练集不变
- 新增跨窗口 LRU 缓存：缓存组级样本(X,y,tickers)，滚动窗口间复用
- 构造样本时一次性清洗与裁剪（非有限值置零、阈值裁剪），固定长度 FIX_T，避免溢出告警
- 训练前一次性标准化整批训练数据；预测对该组一次性标准化
- 使用 DataLoader（pin_memory + 多进程）与 AMP GradScaler(新接口)
- 其余超参（HIDDEN=64、BATCH=8192、EPOCHS等）保持与原设置一致（从 CFG 读取）
"""

import os, h5py, numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from collections import OrderedDict
from tqdm import tqdm, trange

from config import CFG
from backtest_rolling_config import BT_ROLL_CFG
from utils import load_calendar, weekly_fridays
from train_utils import make_filter_fn, load_stock_info, load_flag_table

# ----------------- 设备与超参（保持你的默认） -----------------
device = torch.device(CFG.device)
USE_AMP = device.type == "cuda"  # 仅在 GPU 使用 AMP

SEQ_TRUNC = int(getattr(CFG, "seq_trunc", 60))
FIX_T = int(getattr(CFG, "fix_seq_len", min(SEQ_TRUNC, 60)))  # 固定裁齐长度，默认 60
BATCH = int(getattr(CFG, "batch_size", 8192))                 # 保持 8192
EPOCHS = int(getattr(CFG, "epochs", 4))                       # 保持 4
LR = float(getattr(CFG, "lr", 1e-4))
HIDDEN = int(getattr(CFG, "hidden", 64))                      # 保持 64
LAYERS = int(getattr(CFG, "layers", 1))
DROPOUT = float(getattr(CFG, "dropout", 0.1))
CLIP_NORM = float(getattr(CFG, "clip_norm", 1.0))

# 数据清洗裁剪（避免溢出/NaN）
CLIP_ABS = float(getattr(CFG, "feat_clip_abs", 1e6))

# 跨窗口缓存大小（LRU），可在 CFG.lstm_cache_size 调整
CACHE_SIZE = int(getattr(CFG, "lstm_cache_size", 600))

# DataLoader 并行
NUM_WORKERS = int(getattr(CFG, "num_workers", max(1, (os.cpu_count() or 8) // 4)))

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

TRAIN_YEARS = int(CFG.train_years)  # 5
STEP_WEEKS  = int(CFG.step_weeks)   # 一般 52

# ----------------- 工具 -----------------
def date_to_group(h5: h5py.File) -> Dict[pd.Timestamp, str]:
    rows = []
    for k in h5.keys():
        if not k.startswith("date_"): 
            continue
        d = h5[k].attrs.get("date", None)
        if d is None: 
            continue
        if isinstance(d, bytes): 
            d = d.decode("utf-8")
        rows.append((pd.to_datetime(d), k))
    rows.sort(key=lambda x: x[0])
    return dict(rows)

def map_to_h5_group_date(target_date: pd.Timestamp, h5_idx: pd.DatetimeIndex) -> Optional[pd.Timestamp]:
    ok = h5_idx[h5_idx <= target_date]
    return ok[-1] if len(ok) > 0 else None

def clean_and_clip(X: np.ndarray) -> np.ndarray:
    # 非有限值置零 + 裁剪，避免 nanmean/nanstd 溢出
    X = np.where(np.isfinite(X), X, 0.0)
    if CLIP_ABS is not None and CLIP_ABS > 0:
        X = np.clip(X, -CLIP_ABS, CLIP_ABS)
    return X

def build_group_samples(h5: h5py.File, gk: str, label_df: pd.DataFrame, filter_fn):
    """
    返回：
      X_arr: [M, T, C] float32，已截断至 FIX_T，已清洗/裁剪但未标准化
      y:     [M] float32
      tickers: list[str]
    """
    g = h5[gk]
    d = pd.Timestamp(g.attrs["date"])
    stocks = g["stocks"][:].astype(str)
    X_all = np.asarray(g["factor"][:]); X_all = np.squeeze(X_all)  # [N, T, C] 或 [N, C]
    if X_all.ndim == 2:
        X_all = X_all[:, None, :]
    # 截到最近 SEQ_TRUNC
    if X_all.shape[1] > SEQ_TRUNC:
        X_all = X_all[:, -SEQ_TRUNC:, :]
    # 统一裁齐到 FIX_T（取最近 FIX_T）
    T = min(FIX_T, X_all.shape[1])
    X_all = X_all[:, -T:, :]

    # 标签对齐（MultiIndex 更稳健）
    if isinstance(label_df.index, pd.MultiIndex):
        try:
            sub = label_df.xs(d, level=0)
        except KeyError:
            return np.zeros((0, T, X_all.shape[-1]), dtype=np.float32), np.zeros((0,), dtype=np.float32), []
        sub = sub.reindex(stocks)
        if "next_week_return" not in sub.columns:
            raise KeyError("label_df 缺少列: next_week_return")
        keep_mask = ~sub["next_week_return"].isna().to_numpy()
    else:
        sub = label_df[label_df["date"] == d].set_index("stock").reindex(stocks)
        if "next_week_return" not in sub.columns:
            raise KeyError("label_df 缺少列: next_week_return")
        keep_mask = ~sub["next_week_return"].isna().to_numpy()

    if filter_fn is not None:
        tradable = np.fromiter((filter_fn(d, s) for s in stocks), dtype=bool, count=len(stocks))
        keep_mask &= tradable

    if not keep_mask.any():
        return np.zeros((0, T, X_all.shape[-1]), dtype=np.float32), np.zeros((0,), dtype=np.float32), []

    X_kept = X_all[keep_mask].astype(np.float32, copy=False)
    X_kept = clean_and_clip(X_kept)

    tickers = list(stocks[keep_mask])
    y = sub.loc[keep_mask, "next_week_return"].to_numpy(dtype=np.float32)

    return X_kept, y, tickers

class SeqScaler:
    def __init__(self):
        self.mean=None; self.std=None
    def fit_from_array(self, X_arr: np.ndarray):
        # X_arr: [N, T, C]，展平时间与样本维
        Xf = X_arr.reshape(-1, X_arr.shape[-1])
        self.mean = np.nanmean(Xf, axis=0, keepdims=True)
        self.std  = np.nanstd (Xf, axis=0, keepdims=True) + 1e-6
    def transform(self, x: np.ndarray) -> np.ndarray:
        # x: [N or B, T, C]
        z = (x - self.mean) / self.std
        return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

# ----------------- 模型 -----------------
class LSTMReg(nn.Module):
    def __init__(self, in_dim, hidden=HIDDEN, layers=LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden, num_layers=layers,
                            batch_first=True, dropout=(dropout if layers>1 else 0.0))
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        out, _ = self.lstm(x)     # [B, T, H]
        h = out.mean(dim=1)       # 全局池化
        return self.head(h).squeeze(-1)

def train_epochs(model, opt, scaler_amp, crit, loader, epochs: int):
    model.train()
    for ep in range(1, epochs+1):
        loss_sum, nsum = 0.0, 0
        for xb, yb in loader:
            xb_t = xb.to(device, non_blocking=True)
            yb_t = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            if USE_AMP:
                with torch.amp.autocast(device_type='cuda', enabled=True):
                    pred = model(xb_t)
                    loss = crit(pred, yb_t)
                scaler_amp.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                scaler_amp.step(opt)
                scaler_amp.update()
            else:
                pred = model(xb_t)
                loss = crit(pred, yb_t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                opt.step()
            bs = xb.shape[0]
            loss_sum += float(loss.item()) * bs
            nsum += bs
        yield ep, loss_sum / max(1, nsum)

def predict_group(model, X_arr_t: torch.Tensor, batch_size: int):
    # X_arr_t: torch.FloatTensor [N, T, C] on CPU
    model.eval()
    preds = []
    with torch.no_grad():
        for st in range(0, X_arr_t.shape[0], batch_size):
            xb = X_arr_t[st:st+batch_size].to(device, non_blocking=True)
            if USE_AMP:
                with torch.amp.autocast(device_type='cuda', enabled=True):
                    p = model(xb)
            else:
                p = model(xb)
            preds.append(p.detach().float().cpu().numpy())
    return np.concatenate(preds, axis=0)

# ----------------- LRU 缓存 -----------------
class GroupCacheLRU:
    """
    缓存键：group key (gk)
    值：(X_arr: [M,T,C] float32 (cleaned/clipped), y: [M] float32, tickers: list[str], date: pd.Timestamp)
    限制：最多保存 max_size 个组；超出时弹出最久未用的条目
    """
    def __init__(self, max_size: int = 600):
        self.max_size = max_size
        self.od = OrderedDict()
    def get(self, key):
        v = self.od.get(key)
        if v is not None:
            # LRU 更新
            self.od.move_to_end(key)
        return v
    def put(self, key, value):
        self.od[key] = value
        self.od.move_to_end(key)
        if len(self.od) > self.max_size:
            self.od.popitem(last=False)  # 弹出最久未用
    def __len__(self):
        return len(self.od)

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
        # 推断特征维 C
        C = None
        for k in h5.keys():
            if not k.startswith("date_"): 
                continue
            arr = np.asarray(h5[k]["factor"][:]); arr = np.squeeze(arr)
            C = int(arr.shape[-1]); break
        if C is None:
            raise RuntimeError("无法推断因子维度 C")

        d2g = date_to_group(h5)
        h5_dates = pd.DatetimeIndex(sorted(d2g.keys()))
        preds_all = []

        # 跨窗口缓存
        cache = GroupCacheLRU(max_size=CACHE_SIZE)

        train_weeks = TRAIN_YEARS * 52
        step_weeks  = STEP_WEEKS

        pbar_windows = trange(train_weeks, len(fridays_bt), step_weeks, desc="[LSTM] fast rolling", leave=True)
        for i in pbar_windows:
            train_dates = fridays_bt[i - train_weeks : i]  # 保留完整5年
            pred_dates  = fridays_bt[i : min(i + step_weeks, len(fridays_bt))]

            # 构造训练集（从缓存取或构建）
            X_tr_list, y_tr_list = [], []
            for d in tqdm(train_dates, desc="  build train (fridays)", leave=False):
                g_date = d if d in d2g else map_to_h5_group_date(d, h5_dates)
                if g_date is None: 
                    continue
                gk = d2g[g_date]
                val = cache.get(gk)
                if val is not None:
                    X_arr, ys, _tickers, _dt = val
                else:
                    X_arr, ys, tickers = build_group_samples(h5, gk, label_df, filter_fn)
                    cache.put(gk, (X_arr, ys, tickers, g_date))
                if X_arr.shape[0] == 0: 
                    continue
                X_tr_list.append(X_arr)
                y_tr_list.append(ys)

            if not X_tr_list:
                pbar_windows.write(f"[LSTM] window {i} no train samples, skip")
                continue

            X_train = np.concatenate(X_tr_list, axis=0)  # [N, T, C]
            y_train = np.concatenate(y_tr_list, axis=0)  # [N]

            pbar_windows.write(f"[LSTM] window@{i} train_samples={X_train.shape[0]} T={X_train.shape[1]} C={X_train.shape[2]}")

            # 标准化（一次性）
            seq_scaler = SeqScaler()
            seq_scaler.fit_from_array(X_train)
            X_train = seq_scaler.transform(X_train)  # [N, T, C]

            # DataLoader
            X_train_t = torch.from_numpy(X_train)  # CPU tensor
            y_train_t = torch.from_numpy(y_train)
            train_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X_train_t, y_train_t),
                batch_size=BATCH,
                shuffle=True,
                num_workers=NUM_WORKERS,
                pin_memory=True,
                drop_last=False
            )

            # 模型与优化器（保持你的默认超参）
            model = LSTMReg(in_dim=X_train.shape[-1], hidden=HIDDEN, layers=LAYERS, dropout=DROPOUT).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
            crit = nn.MSELoss()
            scaler_amp = torch.amp.GradScaler("cuda", enabled=USE_AMP)

            # 训练
            best_loss = float("inf")
            for ep, loss in train_epochs(model, opt, scaler_amp, crit, train_loader, EPOCHS):
                pbar_windows.write(f"[LSTM] window@{i} epoch {ep}/{EPOCHS} loss={loss:.6f}")
                if loss + 1e-6 < best_loss:
                    best_loss = loss

            # 预测
            for d in tqdm(pred_dates, desc="  predict (fridays)", leave=False):
                g_date = d if d in d2g else map_to_h5_group_date(d, h5_dates)
                if g_date is None: 
                    continue
                gk = d2g[g_date]
                val = cache.get(gk)
                if val is not None:
                    Xg, yg, tickers, _ = val
                else:
                    Xg, yg, tickers = build_group_samples(h5, gk, label_df, filter_fn)
                    cache.put(gk, (Xg, yg, tickers, g_date))
                if Xg.shape[0] == 0: 
                    continue
                # 标准化
                Xg = seq_scaler.transform(Xg)
                Xg_t = torch.from_numpy(Xg)  # CPU tensor
                pred = predict_group(model, Xg_t, BATCH)
                df_out = pd.DataFrame({
                    "date": d,
                    "stock": np.asarray(tickers, dtype=str),
                    "score": pred.astype(np.float32)
                })
                preds_all.append(df_out)

        if not preds_all:
            raise RuntimeError("无任何LSTM预测输出")
        pred_df = pd.concat(preds_all, ignore_index=True).sort_values(["date","score"], ascending=[True, False])
        out_path = OUT_DIR / f"predictions_filtered_{RUN.run_name_out}.parquet"
        pred_df.to_parquet(out_path)
        print(f"[LSTM] 已保存: {out_path}")

if __name__ == "__main__":
    main()