# ====================== train_utils.py ======================
# coding: utf-8
"""
训练通用工具：
- 环境设置/优化器封装
- 股票样本过滤（读取与闭包）
- AMP/相关指标
- Pairwise Ranking 损失
- 按组读取（HDF5 -> 内存 -> 分batch）
- 窗口内拟合 Scaler（仅用训练组）
"""
import math, torch, numpy as np, pandas as pd, h5py, shutil
import torch.nn.functional as F
from pathlib import Path
from typing import Iterable, Optional, Tuple, List, Dict
from config import CFG
from utils import Scaler, rank_ic
from typing import Any
# ================= RAM 模式（窗口级一次性加载并常驻） =================
def load_window_to_ram(
    h5: h5py.File,
    group_keys: List[str],
    label_df: pd.DataFrame,
    ctx_df: pd.DataFrame,
    scaler_d: Scaler,
    ind_map: dict,
    pad_ind_id: int,
    filter_fn=None
) -> Tuple[List[Dict[str, Any]], float]:
    """
    将一个窗口的所有 group 一次性加载到内存，返回：
    - items: 每个元素为 {date, X, ind, ctx, y}
    - total_gb: 估算总内存占用（GB）
    """
    items: List[Dict[str, Any]] = []
    total_bytes = 0
    skipped = 0
    for gk in group_keys:
        out = load_group_to_memory(h5, gk, label_df, ctx_df, scaler_d, ind_map, pad_ind_id, filter_fn=filter_fn)
        if out is None:
            skipped += 1
            continue
        X, ind, ctx, y, date = out
        items.append({"date": date, "X": X, "ind": ind, "ctx": ctx, "y": y})
        total_bytes += X.nbytes + ind.nbytes + ctx.nbytes + y.nbytes
    total_gb = total_bytes / 1e9
    print(f"[RAM预载] 已加载 {len(items)} 个 group 到内存，约 {total_gb:.2f} GB（跳过空组 {skipped}）")
    return items, total_gb

def iterate_ram_minibatches(mem_items: List[Dict[str, Any]], batch_size: int, shuffle: bool = True):
    """
    仅使用常驻内存的数据进行训练迭代：先打乱 group，再 group 内打乱切 batch。
    """
    if not mem_items:
        return
    order = np.arange(len(mem_items))
    if shuffle:
        np.random.shuffle(order)
    for idx in order:
        it = mem_items[idx]
        X, ind, ctx, y = it["X"], it["ind"], it["ctx"], it["y"]
        M = X.shape[0]
        idxs = np.arange(M)
        if shuffle:
            np.random.shuffle(idxs)
        for st in range(0, M, batch_size):
            sel = idxs[st: st+batch_size]
            xb = torch.from_numpy(X[sel])
            ib = torch.from_numpy(ind[sel])
            cb = torch.from_numpy(ctx[sel])
            yb = torch.from_numpy(y[sel])
            if HAS_CUDA:
                xb = xb.pin_memory(); ib = ib.pin_memory()
                cb = cb.pin_memory(); yb = yb.pin_memory()
            yield xb, ib, cb, yb

# --------------------------- 基础环境/优化器 --------------------------- #
HAS_CUDA = (torch.cuda.is_available() and CFG.device.type == "cuda")

def setup_env():
    if HAS_CUDA and getattr(CFG, "use_tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if HAS_CUDA:
        torch.backends.cudnn.benchmark = True

def try_fused_adamw(params, lr, wd):
    """
    创建 AdamW 优化器：
    - 优先尝试 fused 版本（仅在支持的平台和 PyTorch 版本可用）
    - 使用关键字参数传递 lr 与 weight_decay，避免位置参数被误解析为 betas
    - 回退到非 fused 版本
    """
    kwargs = dict(lr=lr, weight_decay=wd)
    try:
        # 一些 PyTorch 版本要求 fused 仅在 CUDA 上可用
        return torch.optim.AdamW(params, fused=HAS_CUDA, **kwargs)
    except TypeError:
        # 旧版本无 fused 关键字
        return torch.optim.AdamW(params, **kwargs)

def get_amp_dtype():
    if not CFG.use_amp or not HAS_CUDA:
        return None
    s = str(CFG.amp_dtype).lower()
    if s in ("fp16", "float16", "16"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    return None

def ensure_parent_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

# --------------------------- 过滤/数据载入工具 --------------------------- #
def read_csv_with_encoding(path: Path):
    if not Path(path).exists():
        print(f"[筛选] 文件不存在，跳过：{path}")
        return None
    encs = ["utf-8", "gbk", "gb2312", "gb18030", "latin-1"]
    for enc in encs:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"[筛选] 成功用编码 {enc} 读取：{path.name}")
            return df
        except Exception:
            continue
    print(f"[筛选] 无法读取：{path}")
    return None

def parse_date_flexible(s: pd.Series):
    fmts = ['%Y-%m-%d','%Y/%m/%d','%Y%m%d','%d/%m/%Y','%m/%d/%Y']
    for f in fmts:
        try:
            return pd.to_datetime(s, format=f)
        except Exception:
            pass
    try:
        return pd.to_datetime(s, infer_datetime_format=True)
    except Exception:
        return pd.to_datetime(s, errors="coerce")

def load_stock_info(path: Path):
    """
    需要列: [code, ipo_date, delist_date]
    返回: index=code, 列: ipo_date(TS), delist_date(TS or NaT)
    """
    df = read_csv_with_encoding(path)
    if df is None:
        return None
    cols_map = {c.strip().lower(): c for c in df.columns}
    code_col = None
    for cand in ["code","order_book_id","stock_code","ticker"]:
        if cand in cols_map:
            code_col = cols_map[cand]; break
    if code_col is None:
        print("[筛选] stock_info 缺少 code 列，跳过基础信息过滤")
        return None

    ipo_col = None; delist_col = None
    for cand in ["ipo_date","ipo","list_date","上市日期"]:
        l = cand.lower()
        if l in cols_map: ipo_col = cols_map[l]; break
        if cand in cols_map: ipo_col = cols_map[cand]; break
    for cand in ["delist_date","退市日期","退市","de_list_date"]:
        l = cand.lower()
        if l in cols_map: delist_col = cols_map[l]; break
        if cand in cols_map: delist_col = cols_map[cand]; break

    if ipo_col is None:
        print("[筛选] stock_info 缺少 ipo_date 列，跳过基础信息过滤")
        return None

    out = df[[code_col]].copy()
    out[code_col] = out[code_col].astype(str).str.strip()
    out["ipo_date"] = parse_date_flexible(df[ipo_col])
    if delist_col is not None:
        out["delist_date"] = parse_date_flexible(df[delist_col])
        out.loc[out["delist_date"] > pd.Timestamp("2100-01-01"), "delist_date"] = pd.NaT
    else:
        out["delist_date"] = pd.NaT
    out = out.dropna(subset=["ipo_date"])
    return out.set_index(code_col)

def load_flag_table(path: Path):
    """
    停牌/ST 表 -> index=DatetimeIndex(date), 列=股票代码(str)，值 ∈ {0,1}
    """
    df = read_csv_with_encoding(path)
    if df is None:
        return None
    date_col = df.columns[0]
    df["date"] = parse_date_flexible(df[date_col])
    df = df.set_index("date").sort_index()
    for c in df.columns:
        df[c] = df[c].map({"TRUE":1,"FALSE":0,"True":1,"False":0,True:1,False:0,1:1,0:0}).fillna(0).astype(int)
    return df

def code_variants(code: str):
    return [code, code.replace(".XSHE",".XS"), code.replace(".XSHG",".XSH")]

def get_flag_at(df_flag: pd.DataFrame, date: pd.Timestamp, code: str) -> int:
    if df_flag is None or date not in df_flag.index: return 0
    for v in code_variants(code):
        if v in df_flag.columns:
            try: return int(df_flag.at[date, v])
            except Exception: pass
    return 0

def make_filter_fn(daily_df: pd.DataFrame, stock_info: pd.DataFrame, susp_df: pd.DataFrame, st_df: pd.DataFrame):
    """
    返回闭包 filter_fn(date, code) -> bool
    """
    if not getattr(CFG, "enable_filters", False):
        return None
    if not isinstance(daily_df.index, pd.MultiIndex):
        daily_df = daily_df.set_index(["order_book_id","date"]).sort_index()

    def filter_fn(date: pd.Timestamp, code: str) -> bool:
        # IPO/退市
        if stock_info is not None and code in stock_info.index:
            info = stock_info.loc[code]
            ipo = info["ipo_date"]
            if pd.isna(ipo):
                if not CFG.allow_missing_info: return False
            else:
                if (date - ipo).days < CFG.ipo_cut_days: return False
            de = info.get("delist_date", pd.NaT)
            if pd.notna(de) and date >= de: return False
        else:
            if not CFG.allow_missing_info: return False

        # 停牌
        if getattr(CFG, "suspended_exclude", True) and susp_df is not None:
            if get_flag_at(susp_df, date, code) == 1: return False
        # ST
        if getattr(CFG, "st_exclude", True) and st_df is not None:
            if get_flag_at(st_df, date, code) == 1: return False

        # 成交额
        thr = getattr(CFG, "min_daily_turnover", 0.0) or 0.0
        if thr > 0:
            key = (code, date)
            if key not in daily_df.index: return False
            to = daily_df.loc[key, "total_turnover"] if "total_turnover" in daily_df.columns else np.nan
            if pd.isna(to) or float(to) < float(thr): return False

        return True
    return filter_fn

# --------------------------- AMP / 指标 --------------------------- #
def pearsonr(pred, y):
    if pred.numel() < 2:
        return torch.tensor(float("nan"), device=pred.device, dtype=pred.dtype)
    px = pred - pred.mean()
    py = y - y.mean()
    vx = px.pow(2).mean()
    vy = py.pow(2).mean()
    eps = 1e-12
    if vx <= eps or vy <= eps:
        return torch.tensor(float("nan"), device=pred.device, dtype=pred.dtype)
    return (px * py).mean() / (vx.sqrt() * vy.sqrt())

# --------------------------- 排序损失 --------------------------- #
def pairwise_ranking_loss(pred: torch.Tensor, y: torch.Tensor, num_pairs: int = 2048) -> torch.Tensor:
    """
    BPR/Logistic 风格的 pairwise 排序损失：
    L = - E_{(i,j)} [ log sigmoid( (y_i - y_j) * (pred_i - pred_j) ) ]
    - 在 batch 内随机采样若干对，避免 O(B^2)
    """
    B = pred.shape[0]
    if B < 2:
        return pred.new_tensor(0.0)
    device = pred.device
    i = torch.randint(0, B, (num_pairs,), device=device)
    j = torch.randint(0, B, (num_pairs,), device=device)
    mask = (i != j)
    if not mask.any():
        return pred.new_tensor(0.0)
    i = i[mask]; j = j[mask]
    s_ij = torch.sign(y[i] - y[j])  # -1/0/+1
    valid = (s_ij != 0)
    if not valid.any():
        return pred.new_tensor(0.0)
    i = i[valid]; j = j[valid]; s_ij = s_ij[valid]
    diff = (pred[i] - pred[j]) * s_ij
    return -torch.log(torch.sigmoid(diff).clamp_min(1e-12)).mean()

# --------------------------- 窗口内拟合 Scaler --------------------------- #
def fit_scaler_for_groups(h5: h5py.File, group_keys: Iterable[str]) -> Scaler:
    """
    使用训练组的 factor 数据拟合标准化参数，避免未来信息。
    返回 Scaler（其 mean/std 形状 [1,1,C]）。
    """
    scaler = Scaler()
    all_sum = None
    all_sqsum = None
    n_total = 0
    C = None
    for gk in group_keys:
        if gk not in h5: continue
        arr = np.asarray(h5[gk]["factor"][:])
        arr = np.squeeze(arr)  # [N,T,C]
        if arr.ndim != 3: continue
        N, T, C = arr.shape
        x = arr.reshape(-1, C)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        s = np.nansum(x, 0)
        ss = np.nansum(x**2, 0)
        all_sum = s if all_sum is None else all_sum + s
        all_sqsum = ss if all_sqsum is None else all_sqsum + ss
        n_total += x.shape[0]
    if n_total == 0 or C is None:
        scaler.mean = np.zeros((1,1,1), dtype=np.float32)
        scaler.std  = np.ones((1,1,1), dtype=np.float32)
        return scaler
    mean = (all_sum / n_total).reshape(1,1,C)
    var = (all_sqsum / n_total) - (mean.reshape(C)**2)
    std = (np.sqrt(np.maximum(var, 0)) + 1e-6).reshape(1,1,C)
    scaler.mean = mean; scaler.std = std
    return scaler

# --------------------------- HDF5 组读取 -> minibatch --------------------------- #
def load_group_to_memory(h5: h5py.File,
                         gk: str,
                         label_df: pd.DataFrame,
                         ctx_df: pd.DataFrame,
                         scaler_d: Scaler,
                         ind_map: dict,
                         pad_ind_id: int,
                         filter_fn=None):
    """
    读取单个 group（一个周五采样日），返回张量化前的 numpy 批数据。
    """
    if gk not in h5: return None
    g = h5[gk]
    date = pd.Timestamp(g.attrs["date"])
    stocks = g["stocks"][:].astype(str)         # [N]
    X = np.asarray(g["factor"][:])              # [N,T,C] 或 [N,T,1,C]
    X = np.squeeze(X)                           # -> [N,T,C]
    X = scaler_d.transform(X).astype(np.float32)

    # 过滤：标签存在 + 业务过滤
    keep_idx, y_list = [], []
    for i, s in enumerate(stocks):
        key = (date, s)
        if key in label_df.index:
            if (filter_fn is None) or filter_fn(date, s):
                keep_idx.append(i)
                y_list.append(float(label_df.loc[key, "next_week_return"]))
    if not keep_idx:
        return None

    X = X[keep_idx]
    stocks = stocks[keep_idx]
    y = np.asarray(y_list, dtype=np.float32)

    # 行业ID
    ind = np.asarray([ind_map.get(s, pad_ind_id) for s in stocks], dtype=np.int64)

    # 上下文
    if date in ctx_df.index:
        ctx_vec = ctx_df.loc[date].values.astype(np.float32)
    else:
        ctx_vec = np.zeros(ctx_df.shape[1], dtype=np.float32)
    ctx = np.broadcast_to(ctx_vec, (X.shape[0], ctx_vec.shape[0])).copy()

    return X, ind, ctx, y, date

def iterate_group_minibatches(h5: h5py.File,
                              group_keys: List[str],
                              label_df: pd.DataFrame,
                              ctx_df: pd.DataFrame,
                              scaler_d: Scaler,
                              ind_map: dict,
                              pad_ind_id: int,
                              batch_size: int,
                              shuffle: bool = True,
                              filter_fn=None):
    order = np.arange(len(group_keys))
    if shuffle:
        np.random.shuffle(order)
    for idx in order:
        gk = group_keys[idx]
        out = load_group_to_memory(h5, gk, label_df, ctx_df, scaler_d, ind_map, pad_ind_id, filter_fn=filter_fn)
        if out is None: continue
        X, ind, ctx, y, _ = out
        M = X.shape[0]
        idxs = np.arange(M)
        if shuffle:
            np.random.shuffle(idxs)
        for st in range(0, M, batch_size):
            sel = idxs[st:st+batch_size]
            xb = torch.from_numpy(X[sel])
            ib = torch.from_numpy(ind[sel])
            cb = torch.from_numpy(ctx[sel])
            yb = torch.from_numpy(y[sel])
            if HAS_CUDA:
                xb = xb.pin_memory(); ib = ib.pin_memory()
                cb = cb.pin_memory(); yb = yb.pin_memory()
            yield xb, ib, cb, yb