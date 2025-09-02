# ====================== train_rolling.py ======================
# coding: utf-8
"""
滚动训练主循环（日频 + 行业图/关闭）
- 修复 AMP FutureWarning（统一用 torch.amp）
- 极致提速：按 group（每个周五）整块读取 → 内存分 batch 训练（绕开 HDF5 小行随机读瓶颈）
- 实时输出训练指标；按周五分组验证，保存每窗口 best；更新全局最优与最近N最优
- 修复无 CUDA 时 .pin_memory() 报错：仅在 has_cuda=True 时使用
"""
import os, platform, torch, pandas as pd, numpy as np, h5py, shutil, math
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from contextlib import nullcontext
from config import CFG
from utils import pkl_load, rank_ic, weekly_fridays, load_calendar, load_industry_map
from model import GCFNet

# --------------------------- 配置/环境/工具 --------------------------- #
# ============ 新增：通用读取/解析工具（本文件内即可） ============ #
def _read_csv_with_encoding(path: Path):
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

def _parse_date_flexible(s: pd.Series):
    # 常见格式优先，失败则自动推断
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

def _load_stock_info(path: Path):
    """
    需要列: [code, ipo_date, delist_date]（列名大小写/空白会被清洗）
    返回: index=code, 列: ipo_date(TS), delist_date(TS or NaT)
    """
    df = _read_csv_with_encoding(path)
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

    # 宽松匹配 ipo/delist
    ipo_col = None; delist_col = None
    for cand in ["ipo_date","ipo","list_date","上市日期"]:
        l = cand.lower()
        if l in cols_map:
            ipo_col = cols_map[l]; break
        if cand in cols_map:
            ipo_col = cols_map[cand]; break
    for cand in ["delist_date","退市日期","退市","de_list_date"]:
        l = cand.lower()
        if l in cols_map:
            delist_col = cols_map[l]; break
        if cand in cols_map:
            delist_col = cols_map[cand]; break

    if ipo_col is None:
        print("[筛选] stock_info 缺少 ipo_date 列，跳过基础信息过滤")
        return None

    out = df[[code_col]].copy()
    out[code_col] = out[code_col].astype(str).str.strip()
    out["ipo_date"] = _parse_date_flexible(df[ipo_col])
    if delist_col is not None:
        out["delist_date"] = _parse_date_flexible(df[delist_col])
        # 2200-01-01/极大日期 → 视为未退市
        out.loc[out["delist_date"] > pd.Timestamp("2100-01-01"), "delist_date"] = pd.NaT
    else:
        out["delist_date"] = pd.NaT
    out = out.dropna(subset=["ipo_date"])
    return out.set_index(code_col)

def _load_flag_table(path: Path):
    """
    读取停牌/ST 表（行: date, 列: 股票; 值: TRUE/FALSE/1/0）
    返回: index=DatetimeIndex(date), 列=股票代码(str)，值 ∈ {0,1}
    """
    df = _read_csv_with_encoding(path)
    if df is None:
        return None
    # 第一列视为日期
    date_col = df.columns[0]
    df["date"] = _parse_date_flexible(df[date_col])
    df = df.set_index("date").sort_index()
    # 统一 0/1
    for c in df.columns:
        df[c] = df[c].map({"TRUE":1,"FALSE":0,"True":1,"False":0,True:1,False:0,1:1,0:0}).fillna(0).astype(int)
    return df

def _code_variants(code: str):
    # 某些外部文件可能用 .XS / .XSH 简写，这里都尝试
    return [
        code,
        code.replace(".XSHE",".XS"),
        code.replace(".XSHG",".XSH")
    ]

def _get_flag_at(df_flag: pd.DataFrame, date: pd.Timestamp, code: str) -> int:
    if df_flag is None or date not in df_flag.index:
        return 0
    cols = df_flag.columns
    for v in _code_variants(code):
        if v in cols:
            try:
                return int(df_flag.at[date, v])
            except Exception:
                pass
    return 0

def make_filter_fn(
    daily_df: pd.DataFrame,
    stock_info: pd.DataFrame,
    susp_df: pd.DataFrame,
    st_df: pd.DataFrame
):
    """
    返回闭包 filter_fn(date, code) -> bool
    """
    if not getattr(CFG, "enable_filters", False):
        return None

    # 确保 MultiIndex 且名称包含 ('order_book_id','date')
    if not isinstance(daily_df.index, pd.MultiIndex):
        daily_df = daily_df.set_index(["order_book_id","date"]).sort_index()

    def filter_fn(date: pd.Timestamp, code: str) -> bool:
        # 1) 基础信息（IPO/退市）
        if stock_info is not None and code in stock_info.index:
            info = stock_info.loc[code]
            ipo = info["ipo_date"]
            if pd.isna(ipo):
                if not CFG.allow_missing_info:
                    return False
            else:
                days_since_ipo = (date - ipo).days
                if days_since_ipo < CFG.ipo_cut_days:
                    return False
            de = info.get("delist_date", pd.NaT)
            if pd.notna(de) and date >= de:
                return False
        else:
            if not CFG.allow_missing_info:
                return False

        # 2) 停牌
        if getattr(CFG, "suspended_exclude", True) and susp_df is not None:
            if _get_flag_at(susp_df, date, code) == 1:
                return False

        # 3) ST
        if getattr(CFG, "st_exclude", True) and st_df is not None:
            if _get_flag_at(st_df, date, code) == 1:
                return False

        # 4) 成交额
        thr = getattr(CFG, "min_daily_turnover", 0.0) or 0.0
        if thr > 0:
            key = (code, date)
            if key not in daily_df.index:
                return False
            to = daily_df.loc[key, "total_turnover"] if "total_turnover" in daily_df.columns else np.nan
            if pd.isna(to) or float(to) < float(thr):
                return False

        return True

    return filter_fn

# --------------------------- 环境/全局 --------------------------- #
HAS_CUDA = (torch.cuda.is_available() and CFG.device.type == "cuda")

def _get_amp_dtype():
    if not CFG.use_amp or not HAS_CUDA:
        return None
    s = str(CFG.amp_dtype).lower()
    if s in ("fp16", "float16", "16"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    return None

def _pearsonr(pred, y):
    # pred, y: 1D tensor
    if pred.numel() < 2:
        return torch.tensor(float("nan"), device=pred.device, dtype=pred.dtype)
    # 去均值
    px = pred - pred.mean()
    py = y - y.mean()
    vx = px.pow(2).mean()
    vy = py.pow(2).mean()
    # 若任一方方差近似 0，返回 nan（外层会 fall back 到 MSE）
    eps = 1e-12
    if vx <= eps or vy <= eps:
        return torch.tensor(float("nan"), device=pred.device, dtype=pred.dtype)
    cc = (px * py).mean() / (vx.sqrt() * vy.sqrt())
    return cc

def _setup_env():
    if HAS_CUDA and getattr(CFG, "use_tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if HAS_CUDA:
        torch.backends.cudnn.benchmark = True

def _ensure_parent_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def _try_fused_adamw(params, lr, wd):
    try:
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd, fused=HAS_CUDA)
    except TypeError:
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)

# --------------------------- 超快IO：按group整块读取 --------------------------- #
def load_group_to_memory(h5: h5py.File,
                         gk: str,
                         label_df: pd.DataFrame,
                         ctx_df: pd.DataFrame,
                         scaler_d,
                         ind_map: dict,
                         pad_ind_id: int,
                         filter_fn=None):  # 新增
    if gk not in h5:
        return None
    g = h5[gk]
    date = pd.Timestamp(g.attrs["date"])
    stocks = g["stocks"][:].astype(str)                 # [N]
    X = g["factor"][:]                                  # [N,T,C]

    # 标准化
    X = scaler_d.transform(X).astype(np.float32)

    # 过滤无标签 + 业务过滤
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

def iterate_group_minibatches(h5, group_keys, label_df, ctx_df, scaler_d, ind_map, pad_ind_id,
                              batch_size, shuffle=True, filter_fn=None):  # 新增 filter_fn
    order = np.arange(len(group_keys))
    if shuffle:
        np.random.shuffle(order)
    for idx in order:
        gk = group_keys[idx]
        out = load_group_to_memory(h5, gk, label_df, ctx_df, scaler_d, ind_map, pad_ind_id, filter_fn=filter_fn)
        if out is None:
            continue
        X, ind, ctx, y, _ = out
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

# --------------------------- 训练（group-IO + 累积 + 实时指标） --------------------------- #
def train_one_epoch_fast(model, h5, train_gk, label_df, ctx_df, scaler_d, ind_map, pad_ind_id,
                         opt, scaler_grad, amp_dtype, print_interval=10, filter_fn=None):  # 新增 filter_fn
    model.train()
    use_amp = (amp_dtype is not None)
    autocast_ctx = torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp) if use_amp else nullcontext()

    loss_sum = icp_sum = icr_sum = 0.0
    n_sum = 0
    opt.zero_grad(set_to_none=True)

    step = 0
    pbar = tqdm(iterate_group_minibatches(
        h5, train_gk, label_df, ctx_df, scaler_d, ind_map, pad_ind_id,
        batch_size=CFG.batch_size, shuffle=True, filter_fn=filter_fn  # 传入
    ), total=None, leave=False, desc="train-fast")

    for fd, ind, ctx, y in pbar:
        fd = fd.to(CFG.device, non_blocking=HAS_CUDA)
        ind = ind.to(CFG.device, non_blocking=HAS_CUDA)
        ctx = ctx.to(CFG.device, non_blocking=HAS_CUDA)
        y  = y.to(CFG.device, non_blocking=HAS_CUDA)

        with autocast_ctx:
            pred = model(fd, ind, ctx)
            cc = _pearsonr(pred, y)
            loss = (1 - cc) if not torch.isnan(cc) else F.mse_loss(pred, y)

        loss = loss / CFG.grad_accum_steps
        if amp_dtype == torch.float16:
            scaler_grad.scale(loss).backward()
        else:
            loss.backward()

        do_step = ((step + 1) % CFG.grad_accum_steps == 0)
        if do_step:
            if amp_dtype == torch.float16:
                scaler_grad.step(opt); scaler_grad.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)

        # metrics
        with torch.no_grad():
            bs = y.shape[0]
            n_sum += bs
            loss_sum += (loss.detach().float() * CFG.grad_accum_steps).item() * bs
            icp = float(cc.detach().float().item()) if not torch.isnan(cc) else float("nan")
            icr = rank_ic(pred.detach(), y.detach())
            icp_sum += (0.0 if math.isnan(icp) else icp) * bs
            icr_sum += icr * bs

        step += 1
        if (step % max(1, print_interval)) == 0:
            pbar.set_postfix(
                loss=f"{loss_sum/max(1,n_sum):.4f}",
                ic_p=f"{icp_sum/max(1,n_sum):.4f}",
                ic_r=f"{icr_sum/max(1,n_sum):.4f}"
            )

    return {
        "loss": loss_sum / max(1, n_sum),
        "ic_pearson": icp_sum / max(1, n_sum),
        "ic_rank": icr_sum / max(1, n_sum),
        "n": n_sum
    }

# --------------------------- 验证（group-IO） --------------------------- #
@torch.no_grad()
def evaluate_by_groups_fast(model, h5, val_gk, label_df, ctx_df, scaler_d, ind_map, pad_ind_id, filter_fn=None):
    model.eval()
    per_date = []
    for gk in val_gk:
        out = load_group_to_memory(h5, gk, label_df, ctx_df, scaler_d, ind_map, pad_ind_id, filter_fn=filter_fn)
        if out is None:
            continue
        X, ind, ctx, y, _ = out
        X = torch.from_numpy(X)
        ind = torch.from_numpy(ind)
        ctx = torch.from_numpy(ctx)
        y  = torch.from_numpy(y)

        # 送设备
        X = X.to(CFG.device, non_blocking=HAS_CUDA)
        ind = ind.to(CFG.device, non_blocking=HAS_CUDA)
        ctx = ctx.to(CFG.device, non_blocking=HAS_CUDA)
        y  = y.to(CFG.device, non_blocking=HAS_CUDA)

        # 分批避免显存峰值
        preds = []
        bs = CFG.batch_size
        for st in range(0, X.shape[0], bs):
            pred = model(X[st:st+bs], ind[st:st+bs], ctx[st:st+bs])
            preds.append(pred.detach().float().cpu())
        pred = torch.cat(preds, 0)
        ycpu = y.detach().float().cpu()

        mse  = F.mse_loss(pred, ycpu).item()
        cc   = _pearsonr(pred, ycpu)
        cc   = float(cc.item()) if not torch.isnan(cc) else float("nan")
        ric  = rank_ic(pred, ycpu)
        per_date.append({"mse": mse, "ic_pearson": cc, "ic_rank": ric, "n": len(ycpu)})

    if len(per_date) == 0:
        return {"avg_mse": float("nan"), "avg_ic_pearson": float("nan"),
                "avg_ic_rank": float("nan"), "std_ic_rank": float("nan"),
                "dates": 0}

    n_total = sum(d["n"] for d in per_date)
    w = [d["n"] / n_total for d in per_date]
    avg_mse = sum(d["mse"] * w_i for d, w_i in zip(per_date, w))
    avg_icp = sum((0.0 if math.isnan(d["ic_pearson"]) else d["ic_pearson"]) * w_i for d, w_i in zip(per_date, w))
    avg_icr = sum(d["ic_rank"] * w_i for d, w_i in zip(per_date, w))
    std_icr = float(np.std([d["ic_rank"] for d in per_date], ddof=1)) if len(per_date) > 1 else 0.0

    return {"avg_mse": float(avg_mse), "avg_ic_pearson": float(avg_icp),
            "avg_ic_rank": float(avg_icr), "std_ic_rank": float(std_icr),
            "dates": len(per_date)}

# --------------------------- 模型登记/选择 --------------------------- #
def save_checkpoint(model, path: Path):
    _ensure_parent_dir(path)
    torch.save(model.state_dict(), path)

def append_registry(reg_csv: Path, row: dict):
    _ensure_parent_dir(reg_csv)
    df_new = pd.DataFrame([row])
    if reg_csv.exists():
        df_old = pd.read_csv(reg_csv)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(reg_csv, index=False)

def select_global_and_recent(reg_csv: Path, alpha=0.5, recent_topN=5):
    if not reg_csv.exists():
        return None, None
    df = pd.read_csv(reg_csv)
    if len(df) == 0:
        return None, None
    df["score"] = df["val_avg_rankic"] - alpha * df["val_std_rankic"]
    best_overall = df.loc[df["score"].idxmax()].to_dict()
    best_recent = None
    if recent_topN > 0:
        df_recent = df.sort_values("pred_date").tail(recent_topN).copy()
        df_recent["score"] = df_recent["val_avg_rankic"] - alpha * df_recent["val_std_rankic"]
        best_recent = df_recent.loc[df_recent["score"].idxmax()].to_dict()
    return best_overall, best_recent

def copy_as(path_src: Path, path_dst: Path):
    _ensure_parent_dir(path_dst)
    shutil.copy2(str(path_src), str(path_dst))

# --------------------------- 主流程 --------------------------- #
def main():
    _setup_env()

    # 1. 标签 & 交易日
    label_df = pd.read_parquet(CFG.label_file)
    cal = load_calendar(CFG.trading_day_file)
    fridays = weekly_fridays(cal)
    fridays = fridays[(fridays >= pd.Timestamp(CFG.start_date)) &
                      (fridays <= pd.Timestamp(CFG.end_date))]

    # 2. 路径
    daily_h5 = CFG.feat_file
    ctx_file = CFG.processed_dir / "context_features.parquet"
    ctx_df = pd.read_parquet(ctx_file)

    # 3. Scaler 与行业映射
    scaler_d = pkl_load(CFG.scaler_file)
    ind_map  = load_industry_map(CFG.industry_map_file)

    # 4. 维度
    with h5py.File(daily_h5, "r") as h5d:
        d_in = len(h5d.attrs["factor_cols"])
    ctx_dim = ctx_df.shape[1]
    n_ind_known = max(ind_map.values()) + 1
    pad_ind_id = n_ind_known  # 未知行业放最后

    # 5. 模型/优化器/AMP
    amp_dtype = _get_amp_dtype()
    model = GCFNet(
        d_in=d_in, n_ind=n_ind_known, ctx_dim=ctx_dim,
        hidden=CFG.hidden, ind_emb_dim=CFG.ind_emb,
        graph_type=CFG.graph_type, tr_layers=CFG.tr_layers, gat_layers=getattr(CFG, "gat_layers", 1)
    ).to(CFG.device)
    if getattr(CFG, "use_torch_compile", False) and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("已启用 torch.compile")
        except Exception as e:
            print(f"torch.compile 启用失败：{e}")

    opt = _try_fused_adamw(model.parameters(), lr=CFG.lr, wd=CFG.weight_decay)
    scaler_grad = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    # 5.5 载入辅助数据（仅当启用筛选）
    filter_fn = None
    if getattr(CFG, "enable_filters", False):
        # 日频价格（用于成交额过滤）
        daily_price_df = pd.read_parquet(CFG.price_day_file).sort_index()
        # 基础信息/停牌/ST（均为可选，缺失则自动跳过对应过滤）
        stock_info_df = _load_stock_info(CFG.stock_info_file)
        susp_df = _load_flag_table(CFG.is_suspended_file)
        st_df   = _load_flag_table(CFG.is_st_file)
        filter_fn = make_filter_fn(daily_price_df, stock_info_df, susp_df, st_df)
        if filter_fn is not None:
            print("[筛选] 已启用股票样本过滤")
        else:
            print("[筛选] 未启用过滤（或配置关闭）")

    # 6. 滚动窗
    with h5py.File(daily_h5, "r") as h5:
        for i in range(CFG.train_years * 52,
                       len(fridays) - CFG.val_weeks - 1,
                       CFG.step_weeks):
            train_dates = fridays[i - CFG.train_years * 52 : i]
            val_dates   = fridays[i : i + CFG.val_weeks]
            pred_date   = fridays[i + CFG.val_weeks]

            train_gk = [f"date_{d}" for d in range(len(train_dates))]
            val_gk   = [f"date_{d}" for d in range(len(train_dates), len(train_dates)+len(val_dates))]

            # 粗略样本量
            def _count_samples(keys):
                n = 0
                for k in keys:
                    if k in h5:
                        n += len(h5[k]["stocks"])
                return n
            print(f"=== 窗口 {pred_date.strftime('%Y-%m-%d')} === "
                  f"TrainDates={len(train_dates)} ValDates={len(val_dates)} "
                  f"TrainSamples≈{_count_samples(train_gk)} ValSamples≈{_count_samples(val_gk)}")

            # 训练 & 评估
            best_metric = -1e9
            best_epoch  = -1
            best_path   = CFG.model_dir / f"model_best_{pred_date.strftime('%Y%m%d')}.pth"

            for ep in range(CFG.epochs_warm):
                trm = train_one_epoch_fast(
                    model, h5, train_gk, label_df, ctx_df, scaler_d, ind_map, pad_ind_id,
                    opt, scaler_grad, amp_dtype, print_interval=getattr(CFG, "print_step_interval", 10),
                    filter_fn=filter_fn
                )
                print(f"[{pred_date.strftime('%Y-%m-%d')}] "
                      f"epoch {ep+1}/{CFG.epochs_warm} "
                      f"Train: loss={trm['loss']:.4f} ic_p={trm['ic_pearson']:.4f} ic_r={trm['ic_rank']:.4f}")

                valm = evaluate_by_groups_fast(
                    model, h5, val_gk, label_df, ctx_df, scaler_d, ind_map, pad_ind_id,
                    filter_fn=filter_fn
                )
                print(f"  Val: mse={valm['avg_mse']:.6f} "
                      f"ic_p={valm['avg_ic_pearson']:.4f} "
                      f"ic_r={valm['avg_ic_rank']:.4f} "
                      f"ic_r_std={valm['std_ic_rank']:.4f} dates={valm['dates']}")

                sel = valm["avg_ic_rank"] if getattr(CFG, "select_metric", "rankic") == "rankic" else valm["avg_ic_pearson"]
                if math.isfinite(sel) and sel > best_metric:
                    best_metric = sel
                    best_epoch  = ep + 1
                    save_checkpoint(model, best_path)

            last_path = CFG.model_dir / f"model_{pred_date.strftime('%Y%m%d')}.pth"
            save_checkpoint(model, last_path)
            print(f"窗口结束，已保存：last -> {last_path.name} , best(ep={best_epoch}) -> {best_path.name}")

            # 最终验证登记
            best_state = torch.load(best_path, map_location=CFG.device)
            model.load_state_dict(best_state, strict=False)
            final_val = evaluate_by_groups_fast(
                model, h5, val_gk, label_df, ctx_df, scaler_d, ind_map, pad_ind_id,
                filter_fn=filter_fn
            )

            # 记录
            registry_file = getattr(CFG, "registry_file", CFG.model_dir / "model_registry.csv")
            row = {
                "pred_date": pred_date.strftime("%Y-%m-%d"),
                "best_epoch": best_epoch,
                "best_path": str(best_path),
                "last_path": str(last_path),
                "val_avg_mse": final_val["avg_mse"],
                "val_avg_pearson": final_val["avg_ic_pearson"],
                "val_avg_rankic": final_val["avg_ic_rank"],
                "val_std_rankic": final_val["std_ic_rank"],
                "val_dates": final_val["dates"],
            }
            # 写入
            df_new = pd.DataFrame([row])
            if Path(registry_file).exists():
                df_old = pd.read_csv(registry_file)
                df_all = pd.concat([df_old, df_new], ignore_index=True)
            else:
                df_all = df_new
            Path(registry_file).parent.mkdir(parents=True, exist_ok=True)
            df_all.to_csv(registry_file, index=False)

            # 全局/最近N最优
            alpha = getattr(CFG, "score_alpha", 0.5)
            recentN = getattr(CFG, "recent_topN", 5)
            df = df_all.copy()
            df["score"] = df["val_avg_rankic"] - alpha * df["val_std_rankic"]
            best_overall = df.loc[df["score"].idxmax()].to_dict()
            best_recent = None
            if recentN > 0 and len(df) >= 1:
                df_recent = df.sort_values("pred_date").tail(recentN).copy()
                df_recent["score"] = df_recent["val_avg_rankic"] - alpha * df_recent["val_std_rankic"]
                best_recent = df_recent.loc[df_recent["score"].idxmax()].to_dict()

            if best_overall is not None:
                dst = CFG.model_dir / "best_overall.pth"
                _ensure_parent_dir(dst)
                shutil.copy2(best_overall["best_path"], dst)
            if best_recent is not None:
                dst = CFG.model_dir / f"best_recent_{recentN}.pth"
                _ensure_parent_dir(dst)
                shutil.copy2(best_recent["best_path"], dst)

    print("训练完成。全局最优模型：", (CFG.model_dir / "best_overall.pth").as_posix())

if __name__ == "__main__":
    main()