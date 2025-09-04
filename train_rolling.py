# ====================== train_rolling.py ======================
# coding: utf-8
"""
滚动训练主循环（日频 + 行业图/可关闭）
本版本特性：
- 使用 train_utils 抽离通用函数，精简主文件
- 每个训练窗口仅用“训练分组”拟合 Scaler（避免未来信息泄漏）
- 训练损失：loss = w*(1 - Pearson) + (1 - w)*PairwiseRanking（w=CFG.ranking_weight，默认0.5）
- 训练进度条与日志同时显示：
  * 总损失 loss（权重后的总损失）
  * pairwise 排序损失 pairwise ranking loss（未加权，仅原始 pairwise ranking loss）
  * ic_p（Pearson相关系数）
  * ic_r（RankIC）
- 评估阶段按周五分组计算 MSE / Pearson / RankIC 及 RankIC 的标准差，并输出 pairwise ranking loss 与加权 loss
- 按验证集指标选择 best 模型，登记全局与最近N窗口最优并复制到固定路径
- 新增：RAM 常驻加速路径（窗口级一次性加载 Train/Val 到内存，epoch 内零 IO），由 CFG.ram_accel_enable 控制；
       若估算内存超过 CFG.ram_accel_mem_cap_gb 则自动回退为逐组加载
"""

import math, torch, pandas as pd, numpy as np, h5py, shutil
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from contextlib import nullcontext

from config import CFG
from model import GCFNet
from utils import weekly_fridays, load_calendar, load_industry_map, rank_ic
from train_utils import (
    HAS_CUDA, setup_env, get_amp_dtype, try_fused_adamw, ensure_parent_dir,
    make_filter_fn, load_stock_info, load_flag_table,
    fit_scaler_for_groups, iterate_group_minibatches, load_group_to_memory,
    pairwise_ranking_loss, pearsonr,
    # RAM 模式
    load_window_to_ram, iterate_ram_minibatches
)

# --------------------------- 辅助：保存模型 --------------------------- #
def save_checkpoint(model, path: Path):
    ensure_parent_dir(path)
    torch.save(model.state_dict(), path)

# --------------------------- 主流程 --------------------------- #
def main():
    # 环境开关（TF32、cudnn.benchmark 等）
    setup_env()

    # 1) 标签与交易日历
    label_df = pd.read_parquet(CFG.label_file)
    cal = load_calendar(CFG.trading_day_file)
    fridays = weekly_fridays(cal)
    fridays = fridays[(fridays >= pd.Timestamp(CFG.start_date)) &
                      (fridays <= pd.Timestamp(CFG.end_date))]

    # 2) 特征与上下文
    daily_h5 = CFG.feat_file
    ctx_file = CFG.processed_dir / "context_features.parquet"
    ctx_df = pd.read_parquet(ctx_file)

    # 3) 行业映射与维度
    ind_map  = load_industry_map(CFG.industry_map_file)
    with h5py.File(daily_h5, "r") as h5d:
        d_in = len(h5d.attrs["factor_cols"])
    ctx_dim = ctx_df.shape[1]
    n_ind_known = max(ind_map.values()) + 1
    pad_ind_id = n_ind_known  # 未知行业放最后

    # 4) 构建模型、优化器、AMP
    amp_dtype = get_amp_dtype()
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

    # 重要：AdamW 使用关键字传入 lr 与 weight_decay；fused 在不支持时自动回退
    opt = try_fused_adamw(model.parameters(), CFG.lr, CFG.weight_decay)
    scaler_grad = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    # 5) 样本过滤（可选）
    filter_fn = None
    if getattr(CFG, "enable_filters", False):
        daily_price_df = pd.read_parquet(CFG.price_day_file).sort_index()
        stock_info_df = load_stock_info(CFG.stock_info_file)
        susp_df = load_flag_table(CFG.is_suspended_file)
        st_df   = load_flag_table(CFG.is_st_file)
        filter_fn = make_filter_fn(daily_price_df, stock_info_df, susp_df, st_df)
        print("[筛选] 已启用股票样本过滤" if filter_fn else "[筛选] 未启用过滤（或配置关闭）")

    # 6) 滚动训练
    with h5py.File(daily_h5, "r") as h5:
        # 以“训练年数 * 52周”为滑动窗口的训练起点；每次步进 step_weeks
        for i in range(CFG.train_years * 52,
                       len(fridays) - CFG.val_weeks - 1,
                       CFG.step_weeks):

            # 时间切片
            train_dates = fridays[i - CFG.train_years * 52 : i]
            val_dates   = fridays[i : i + CFG.val_weeks]
            pred_date   = fridays[i + CFG.val_weeks]

            # 将“当前窗口内的周五索引”映射到 HDF5 group key（注意：feature_engineering 写入顺序与 fridays 对齐）
            train_gk = [f"date_{d}" for d in range(len(train_dates))]
            val_gk   = [f"date_{d}" for d in range(len(train_dates), len(train_dates)+len(val_dates))]

            # 仅用“训练组”拟合当期标准化 Scaler（避免未来信息）
            scaler_d = fit_scaler_for_groups(h5, train_gk)

            # 粗略样本量统计
            def _count_samples(keys):
                n = 0
                for k in keys:
                    if k in h5:
                        n += len(h5[k]["stocks"])
                return n
            print(f"=== 窗口 {pred_date.strftime('%Y-%m-%d')} === "
                  f"TrainDates={len(train_dates)} ValDates={len(val_dates)} "
                  f"TrainSamples≈{_count_samples(train_gk)} ValSamples≈{_count_samples(val_gk)}")

            # 选择指标：rankic 或 pearson
            select_metric = getattr(CFG, "select_metric", "rankic")
            w_loss = float(getattr(CFG, "ranking_weight", 0.5))

            # RAM 分支：根据配置决定是否一次性将 Train/Val 载入内存
            use_ram = bool(getattr(CFG, "ram_accel_enable", False))
            mem_items_train = None
            mem_items_val = None
            if use_ram:
                mem_items_train, gb_train = load_window_to_ram(
                    h5, train_gk, label_df, ctx_df, scaler_d, ind_map, pad_ind_id, filter_fn=filter_fn
                )
                mem_items_val, gb_val = load_window_to_ram(
                    h5, val_gk, label_df, ctx_df, scaler_d, ind_map, pad_ind_id, filter_fn=filter_fn
                )
                total_gb = gb_train + gb_val
                cap_gb = float(getattr(CFG, "ram_accel_mem_cap_gb", 48))
                if total_gb > cap_gb:
                    print(f"[RAM预载] 估算内存 {total_gb:.2f} GB 超过上限 {cap_gb:.2f} GB，回退为逐组加载模式")
                    use_ram = False
                    mem_items_train = None
                    mem_items_val = None
                else:
                    print(f"[RAM预载] 启用 RAM 模式，总内存估算 {total_gb:.2f} GB（<= {cap_gb:.2f} GB）")

            # 训练多个 epoch，保存 best
            best_metric = -1e9
            best_epoch  = -1
            best_path   = CFG.model_dir / f"model_best_{pred_date.strftime('%Y%m%d')}.pth"

            for ep in range(CFG.epochs_warm):
                # ---------------- 训练 1 个 epoch ---------------- #
                model.train()
                loss_sum = icp_sum = icr_sum = 0.0
                rank_sum = 0.0  # 额外：统计 pairwise ranking loss 的平均值
                n_sum = 0
                opt.zero_grad(set_to_none=True)
                step = 0

                # 数据迭代器：RAM 常驻或逐组加载
                if use_ram:
                    data_iter = iterate_ram_minibatches(mem_items_train, CFG.batch_size, shuffle=True)
                    pbar = tqdm(data_iter, total=None, leave=False, desc="train-fast[RAM]")
                else:
                    pbar = tqdm(
                        iterate_group_minibatches(
                            h5, train_gk, label_df, ctx_df, scaler_d, ind_map, pad_ind_id,
                            batch_size=CFG.batch_size, shuffle=True, filter_fn=filter_fn
                        ),
                        total=None, leave=False, desc="train-fast"
                    )

                # autocast 上下文（在 amp_dtype 不为 None 且启用 CUDA 时启用）
                autocast_ctx = torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(amp_dtype is not None)) \
                               if (amp_dtype is not None) else nullcontext()

                for fd, ind, ctx, y in pbar:
                    # 送设备
                    fd = fd.to(CFG.device, non_blocking=HAS_CUDA)
                    ind = ind.to(CFG.device, non_blocking=HAS_CUDA)
                    ctx = ctx.to(CFG.device, non_blocking=HAS_CUDA)
                    y  = y.to(CFG.device, non_blocking=HAS_CUDA)

                    with autocast_ctx:
                        pred = model(fd, ind, ctx)
                        # Pearson 部分：若无法计算（方差≈0），回退到 MSE
                        cc = pearsonr(pred, y)
                        loss_pearson = (1 - cc) if not torch.isnan(cc) else F.mse_loss(pred, y)
                        # Pairwise 排序损失（原始未缩放值，用于日志与组合损失）
                        rank_val = pairwise_ranking_loss(pred, y, num_pairs=2048)
                        # 加权总损失
                        loss = w_loss * loss_pearson + (1 - w_loss) * rank_val

                    # 梯度累积：先缩放后 backward
                    loss_scaled = loss / CFG.grad_accum_steps
                    if amp_dtype == torch.float16:
                        scaler_grad.scale(loss_scaled).backward()
                    else:
                        loss_scaled.backward()

                    # 到达累积步数，执行优化器 step
                    do_step = ((step + 1) % CFG.grad_accum_steps == 0)
                    if do_step:
                        if amp_dtype == torch.float16:
                            scaler_grad.step(opt); scaler_grad.update()
                        else:
                            opt.step()
                        opt.zero_grad(set_to_none=True)

                    # 训练中实时统计
                    with torch.no_grad():
                        bs = y.shape[0]
                        n_sum += bs
                        # 注意：日志统计使用“未缩放”的 loss 与 rank_val
                        loss_sum += loss.detach().float().item() * bs
                        rank_sum += rank_val.detach().float().item() * bs
                        icp = float(cc.detach().float().item()) if not torch.isnan(cc) else float("nan")
                        icr = rank_ic(pred.detach(), y.detach())
                        icp_sum += (0.0 if math.isnan(icp) else icp) * bs
                        icr_sum += icr * bs

                    step += 1
                    if (step % max(1, getattr(CFG, "print_step_interval", 10))) == 0:
                        pbar.set_postfix(
                            loss=f"{loss_sum/max(1,n_sum):.4f}",
                            pairwise_ranking_loss=f"{rank_sum/max(1,n_sum):.4f}",
                            ic_p=f"{icp_sum/max(1,n_sum):.4f}",
                            ic_r=f"{icr_sum/max(1,n_sum):.4f}",
                            w=w_loss
                        )

                print(f"[{pred_date.strftime('%Y-%m-%d')}] "
                      f"epoch {ep+1}/{CFG.epochs_warm} "
                      f"Train: loss={loss_sum/max(1,n_sum):.4f} "
                      f"pairwise ranking loss={rank_sum/max(1,n_sum):.4f} "
                      f"ic_p={icp_sum/max(1,n_sum):.4f} ic_r={icr_sum/max(1,n_sum):.4f} "
                      f"w={w_loss}")

                # ---------------- 验证（按组聚合） ---------------- #
                model.eval()
                per_date = []
                if use_ram:
                    # RAM 模式：使用内存中的 val items
                    for it in mem_items_val:
                        X = torch.from_numpy(it["X"]).to(CFG.device, non_blocking=HAS_CUDA)
                        ind_t = torch.from_numpy(it["ind"]).to(CFG.device, non_blocking=HAS_CUDA)
                        ctx_t = torch.from_numpy(it["ctx"]).to(CFG.device, non_blocking=HAS_CUDA)
                        y_t  = torch.from_numpy(it["y"]).to(CFG.device, non_blocking=HAS_CUDA)

                        preds = []
                        bs = CFG.batch_size
                        for st in range(0, X.shape[0], bs):
                            p = model(X[st:st+bs], ind_t[st:st+bs], ctx_t[st:st+bs])
                            preds.append(p.detach().float().cpu())
                        pred_cpu = torch.cat(preds, 0)
                        y_cpu = y_t.detach().float().cpu()

                        mse  = F.mse_loss(pred_cpu, y_cpu).item()
                        cc_t = pearsonr(pred_cpu, y_cpu)
                        cc   = float(cc_t.item()) if not torch.isnan(cc_t) else float("nan")
                        ric  = rank_ic(pred_cpu, y_cpu)
                        prl  = float(pairwise_ranking_loss(pred_cpu, y_cpu, num_pairs=2048).item())
                        if not math.isnan(cc):
                            loss_val = float((w_loss * (1 - cc) + (1 - w_loss) * prl))
                        else:
                            loss_val = float((w_loss * F.mse_loss(pred_cpu, y_cpu).item() + (1 - w_loss) * prl))
                        per_date.append({
                            "mse": mse, "ic_pearson": cc, "ic_rank": ric,
                            "pairwise_rank": prl, "loss": loss_val, "n": len(y_cpu)
                        })
                else:
                    # 原逐组加载
                    for gk in val_gk:
                        out = load_group_to_memory(h5, gk, label_df, ctx_df, scaler_d, ind_map, pad_ind_id, filter_fn=filter_fn)
                        if out is None:
                            continue
                        X, ind_t, ctx_t, y_t = [torch.from_numpy(o) for o in out[:4]]
                        X = X.to(CFG.device, non_blocking=HAS_CUDA)
                        ind_t = ind_t.to(CFG.device, non_blocking=HAS_CUDA)
                        ctx_t = ctx_t.to(CFG.device, non_blocking=HAS_CUDA)
                        y_t  = y_t.to(CFG.device, non_blocking=HAS_CUDA)

                        preds = []
                        bs = CFG.batch_size
                        for st in range(0, X.shape[0], bs):
                            p = model(X[st:st+bs], ind_t[st:st+bs], ctx_t[st:st+bs])
                            preds.append(p.detach().float().cpu())
                        pred_cpu = torch.cat(preds, 0)
                        y_cpu = y_t.detach().float().cpu()

                        mse  = F.mse_loss(pred_cpu, y_cpu).item()
                        cc_t = pearsonr(pred_cpu, y_cpu)
                        cc   = float(cc_t.item()) if not torch.isnan(cc_t) else float("nan")
                        ric  = rank_ic(pred_cpu, y_cpu)
                        prl  = float(pairwise_ranking_loss(pred_cpu, y_cpu, num_pairs=2048).item())
                        if not math.isnan(cc):
                            loss_val = float((w_loss * (1 - cc) + (1 - w_loss) * prl))
                        else:
                            loss_val = float((w_loss * F.mse_loss(pred_cpu, y_cpu).item() + (1 - w_loss) * prl))
                        per_date.append({
                            "mse": mse, "ic_pearson": cc, "ic_rank": ric,
                            "pairwise_rank": prl, "loss": loss_val, "n": len(y_cpu)
                        })

                if len(per_date) == 0:
                    valm = {"avg_mse": float("nan"), "avg_ic_pearson": float("nan"),
                            "avg_ic_rank": float("nan"), "std_ic_rank": float("nan"),
                            "avg_pairwise_rank": float("nan"), "avg_loss": float("nan"),
                            "dates": 0}
                else:
                    n_total = sum(d["n"] for d in per_date)
                    wv = [d["n"] / n_total for d in per_date]
                    avg_mse = sum(d["mse"] * wi for d, wi in zip(per_date, wv))
                    avg_icp = sum((0.0 if math.isnan(d["ic_pearson"]) else d["ic_pearson"]) * wi for d, wi in zip(per_date, wv))
                    avg_icr = sum(d["ic_rank"] * wi for d, wi in zip(per_date, wv))
                    std_icr = float(np.std([d["ic_rank"] for d in per_date], ddof=1)) if len(per_date) > 1 else 0.0
                    avg_prl = sum(d["pairwise_rank"] * wi for d, wi in zip(per_date, wv))
                    avg_loss = sum(d["loss"] * wi for d, wi in zip(per_date, wv))
                    valm = {"avg_mse": float(avg_mse), "avg_ic_pearson": float(avg_icp),
                            "avg_ic_rank": float(avg_icr), "std_ic_rank": float(std_icr),
                            "avg_pairwise_rank": float(avg_prl), "avg_loss": float(avg_loss),
                            "dates": len(per_date)}

                print(f"  Val: mse={valm['avg_mse']:.6f} "
                      f"ic_p={valm['avg_ic_pearson']:.4f} "
                      f"ic_r={valm['avg_ic_rank']:.4f} "
                      f"pairwise ranking loss={valm['avg_pairwise_rank']:.4f} "
                      f"loss={valm['avg_loss']:.4f} "
                      f"ic_r_std={valm['std_ic_rank']:.4f} dates={valm['dates']} "
                      f"w={w_loss}")

                # 保存 best（选择指标支持 rankic 或 pearson）
                sel = valm["avg_ic_rank"] if select_metric == "rankic" else valm["avg_ic_pearson"]
                if math.isfinite(sel) and sel > best_metric:
                    best_metric = sel
                    best_epoch  = ep + 1
                    save_checkpoint(model, best_path)

            # 训练完一个窗口：保存最后一版
            last_path = CFG.model_dir / f"model_{pred_date.strftime('%Y%m%d')}.pth"
            save_checkpoint(model, last_path)
            print(f"窗口结束，已保存：last -> {last_path.name} , best(ep={best_epoch}) -> {best_path.name}")

            # ---------------- 记录与选择全局/最近N最优 ---------------- #
            # 载入 best，统一用同一 scaler_d 在验证集上复算一次指标并登记
            best_state = torch.load(best_path, map_location=CFG.device)
            model.load_state_dict(best_state, strict=False)

            model.eval()
            per_date = []
            if use_ram:
                for it in mem_items_val:
                    X = torch.from_numpy(it["X"]).to(CFG.device, non_blocking=HAS_CUDA)
                    ind_t = torch.from_numpy(it["ind"]).to(CFG.device, non_blocking=HAS_CUDA)
                    ctx_t = torch.from_numpy(it["ctx"]).to(CFG.device, non_blocking=HAS_CUDA)
                    y_t  = torch.from_numpy(it["y"]).to(CFG.device, non_blocking=HAS_CUDA)
                    preds = []
                    bs = CFG.batch_size
                    for st in range(0, X.shape[0], bs):
                        p = model(X[st:st+bs], ind_t[st:st+bs], ctx_t[st:st+bs])
                        preds.append(p.detach().float().cpu())
                    pred_cpu = torch.cat(preds, 0)
                    y_cpu = y_t.detach().float().cpu()
                    mse  = F.mse_loss(pred_cpu, y_cpu).item()
                    cc_t = pearsonr(pred_cpu, y_cpu)
                    cc   = float(cc_t.item()) if not torch.isnan(cc_t) else float("nan")
                    ric  = rank_ic(pred_cpu, y_cpu)
                    prl  = float(pairwise_ranking_loss(pred_cpu, y_cpu, num_pairs=2048).item())
                    if not math.isnan(cc):
                        loss_val = float((w_loss * (1 - cc) + (1 - w_loss) * prl))
                    else:
                        loss_val = float((w_loss * F.mse_loss(pred_cpu, y_cpu).item() + (1 - w_loss) * prl))
                    per_date.append({"mse": mse, "ic_pearson": cc, "ic_rank": ric,
                                     "pairwise_rank": prl, "loss": loss_val, "n": len(y_cpu)})
            else:
                for gk in val_gk:
                    out = load_group_to_memory(h5, gk, label_df, ctx_df, scaler_d, ind_map, pad_ind_id, filter_fn=filter_fn)
                    if out is None: continue
                    X, ind_t, ctx_t, y_t = [torch.from_numpy(o) for o in out[:4]]
                    X = X.to(CFG.device, non_blocking=HAS_CUDA)
                    ind_t = ind_t.to(CFG.device, non_blocking=HAS_CUDA)
                    ctx_t = ctx_t.to(CFG.device, non_blocking=HAS_CUDA)
                    y_t  = y_t.to(CFG.device, non_blocking=HAS_CUDA)
                    preds = []
                    bs = CFG.batch_size
                    for st in range(0, X.shape[0], bs):
                        p = model(X[st:st+bs], ind_t[st:st+bs], ctx_t[st:st+bs])
                        preds.append(p.detach().float().cpu())
                    pred_cpu = torch.cat(preds, 0)
                    y_cpu = y_t.detach().float().cpu()
                    mse  = F.mse_loss(pred_cpu, y_cpu).item()
                    cc_t = pearsonr(pred_cpu, y_cpu)
                    cc   = float(cc_t.item()) if not torch.isnan(cc_t) else float("nan")
                    ric  = rank_ic(pred_cpu, y_cpu)
                    prl  = float(pairwise_ranking_loss(pred_cpu, y_cpu, num_pairs=2048).item())
                    if not math.isnan(cc):
                        loss_val = float((w_loss * (1 - cc) + (1 - w_loss) * prl))
                    else:
                        loss_val = float((w_loss * F.mse_loss(pred_cpu, y_cpu).item() + (1 - w_loss) * prl))
                    per_date.append({"mse": mse, "ic_pearson": cc, "ic_rank": ric,
                                     "pairwise_rank": prl, "loss": loss_val, "n": len(y_cpu)})

            if len(per_date) == 0:
                final_val = {"avg_mse": float("nan"), "avg_ic_pearson": float("nan"),
                             "avg_ic_rank": float("nan"), "std_ic_rank": float("nan"),
                             "avg_pairwise_rank": float("nan"), "avg_loss": float("nan"),
                             "dates": 0}
            else:
                n_total = sum(d["n"] for d in per_date)
                wv = [d["n"] / n_total for d in per_date]
                avg_mse = sum(d["mse"] * wi for d, wi in zip(per_date, wv))
                avg_icp = sum((0.0 if math.isnan(d["ic_pearson"]) else d["ic_pearson"]) * wi for d, wi in zip(per_date, wv))
                avg_icr = sum(d["ic_rank"] * wi for d, wi in zip(per_date, wv))
                std_icr = float(np.std([d["ic_rank"] for d in per_date], ddof=1)) if len(per_date) > 1 else 0.0
                avg_prl = sum(d["pairwise_rank"] * wi for d, wi in zip(per_date, wv))
                avg_loss = sum(d["loss"] * wi for d, wi in zip(per_date, wv))
                final_val = {"avg_mse": float(avg_mse), "avg_ic_pearson": float(avg_icp),
                             "avg_ic_rank": float(avg_icr), "std_ic_rank": float(std_icr),
                             "avg_pairwise_rank": float(avg_prl), "avg_loss": float(avg_loss),
                             "dates": len(per_date)}

            # 写入登记表
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
                "val_avg_pairwise_ranking_loss": final_val["avg_pairwise_rank"],
                "val_avg_loss": final_val["avg_loss"],
                "val_dates": final_val["dates"],
                "ranking_weight": w_loss,  # 记录当前窗口使用的 w
            }
            df_new = pd.DataFrame([row])
            if Path(registry_file).exists():
                df_old = pd.read_csv(registry_file)
                df_all = pd.concat([df_old, df_new], ignore_index=True)
            else:
                df_all = df_new
            Path(registry_file).parent.mkdir(parents=True, exist_ok=True)
            df_all.to_csv(registry_file, index=False)

            # 选择全局最优与最近N窗口最优，并复制到固定路径
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
                ensure_parent_dir(dst)
                shutil.copy2(best_overall["best_path"], dst)
            if best_recent is not None:
                dst = CFG.model_dir / f"best_recent_{recentN}.pth"
                ensure_parent_dir(dst)
                shutil.copy2(best_recent["best_path"], dst)

    print("训练完成。全局最优模型：", (CFG.model_dir / "best_overall.pth").as_posix())

if __name__ == "__main__":
    main()