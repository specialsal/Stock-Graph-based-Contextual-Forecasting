# ====================== train_rolling.py ======================
# coding: utf-8
"""
滚动训练主循环（日频 + 行业图/可关闭）— 验证集早停版（取消测试集，支持 Warm Start）
要点：
- 仅使用训练集与验证集（取消测试集切分与评估）；
- 每个 epoch 结束后评估验证集：以验证集 avg_ic_rank（越大越好）作为早停与 best 判据；
- best/last 模型保存日期 pred_date = fridays[i + val_weeks]（与窗口验证终点对齐）；
- 去除“风险调整分数 score / alpha / test_weeks”等逻辑与参数；
- 新增 Warm Start：可选从“上一个窗口”的 best 作为本窗口初始化；
- 写入登记表仅记录：best_epoch、val_avg_pearson、val_avg_rankic、val_std_rankic、
  val_avg_pairwise_ranking_loss、val_avg_loss、epoch_count；
- 保留中文注释。
"""

import math, torch, pandas as pd, numpy as np, h5py
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from contextlib import nullcontext
from torch.utils.tensorboard import SummaryWriter
import time
import os

from config import CFG
from model import GCFNet
from utils import weekly_fridays, load_calendar, load_industry_map, rank_ic
from train_utils import (
    HAS_CUDA, setup_env, get_amp_dtype, try_fused_adamw, ensure_parent_dir,
    make_filter_fn, load_stock_info, load_flag_table,
    fit_scaler_for_groups, iterate_group_minibatches, load_group_to_memory,
    pairwise_ranking_loss, pearsonr,
    load_window_to_ram, iterate_ram_minibatches
)

# --------------------------- 辅助：保存模型 --------------------------- #
def save_checkpoint(model, path: Path, scaler_d=None):
    ensure_parent_dir(path)
    payload = {"state_dict": model.state_dict()}
    if scaler_d is not None and getattr(scaler_d, "mean", None) is not None and getattr(scaler_d, "std", None) is not None:
        payload["scaler_mean"] = scaler_d.mean.astype("float32")
        payload["scaler_std"]  = scaler_d.std.astype("float32")
    torch.save(payload, path)

def build_date_to_group_map(h5: h5py.File) -> dict:
    d2g = {}
    for k in h5.keys():
        if not k.startswith("date_"):
            continue
        d = h5[k].attrs.get("date", None)
        if d is None:
            continue
        if isinstance(d, bytes):
            d = d.decode("utf-8")
        try:
            dt = pd.to_datetime(d)
        except Exception:
            continue
        d2g[dt] = k
    return d2g

# --------------------------- 主流程 --------------------------- #
def main():
    setup_env()

    # TensorBoard（统一目录）
    log_root = Path("./models") / "tblogs"
    log_root.mkdir(parents=True, exist_ok=True)
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=log_root / f"run_{run_tag}")

    global_step_step = 0
    global_step_epoch = 0

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
    pad_ind_id = n_ind_known

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

    train_weeks = CFG.train_years * 52
    val_weeks   = int(CFG.val_weeks)
    step_weeks  = int(CFG.step_weeks)
    w_loss      = float(getattr(CFG, "ranking_weight", 0.5))

    out_model_dir = Path(CFG.model_dir)
    out_model_dir.mkdir(parents=True, exist_ok=True)
    registry_file = Path(CFG.registry_file)

    # 6) 滚动训练（仅 Train/Val）
    with h5py.File(daily_h5, "r") as h5:
        date2gk_all = build_date_to_group_map(h5)
        if len(date2gk_all) == 0:
            writer.close()
            raise RuntimeError(f"H5 中没有任何以 date_* 命名的组：{daily_h5}")

        for i in range(train_weeks, len(fridays) - val_weeks, step_weeks):
            # 时间切片
            train_dates = fridays[i - train_weeks : i]
            val_dates   = fridays[i : i + val_weeks]
            pred_date   = fridays[i + val_weeks]

            train_gk = [date2gk_all[d] for d in train_dates if d in date2gk_all]
            val_gk   = [date2gk_all[d] for d in val_dates   if d in date2gk_all]

            # 仅用训练组拟合标准化
            scaler_d = fit_scaler_for_groups(h5, train_gk)

            # -------- Warm Start：尝试载入上一个窗口的 best --------
            if getattr(CFG, "warm_start_enable", False):
                # 上一个窗口的 pred_date_prev = fridays[i - step_weeks + val_weeks]
                if (i - step_weeks) >= train_weeks:
                    pred_date_prev = fridays[i - step_weeks + val_weeks]
                    prev_best_path = out_model_dir / f"model_best_{pred_date_prev.strftime('%Y%m%d')}.pth"
                    if prev_best_path.exists():
                        try:
                            payload = torch.load(prev_best_path, map_location=CFG.device, weights_only=False)
                            state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
                            model.load_state_dict(state_dict, strict=bool(getattr(CFG, "warm_start_strict", False)))
                            print(f"[WarmStart] 已从上一窗口 best 加载初始化：{prev_best_path.name}")
                        except Exception as e:
                            print(f"[WarmStart][警告] 加载上一窗口 best 失败，使用随机初始化。原因：{e}")
                    else:
                        print(f"[WarmStart][提示] 未找到上一窗口 best：{prev_best_path.name}，使用随机初始化。")

            # 样本计数打印
            def _count_samples(keys):
                n = 0
                for k in keys:
                    if k in h5:
                        n += len(h5[k]["stocks"])
                return n

            print(f"=== 窗口 {pred_date.strftime('%Y-%m-%d')} === "
                  f"TrainDates={len(train_dates)}(H5命中={len(train_gk)}) "
                  f"ValDates={len(val_dates)}(H5命中={len(val_gk)}) "
                  f"TrainSamples≈{_count_samples(train_gk)} "
                  f"ValSamples≈{_count_samples(val_gk)}")

            if len(train_gk) == 0 or len(val_gk) == 0:
                print("[警告] 该窗口在 H5 中命中组过少（Train/Val 存在空），跳过本窗口。")
                continue

            # RAM 预载
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

            tb_prefix = f"{pred_date.strftime('%Y%m%d')}"
            window_step = 0

            best_val_score = -1e9
            best_epoch  = -1
            best_path   = out_model_dir / f"model_best_{pred_date.strftime('%Y%m%d')}.pth"

            epochs_no_improve = 0
            epoch_id = 0

            while True:
                epoch_id += 1
                # ---------------- 训练 1 个 epoch ---------------- #
                model.train()
                loss_sum = icp_sum = icr_sum = 0.0
                rank_sum = 0.0
                n_sum = 0
                opt.zero_grad(set_to_none=True)
                step = 0

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

                autocast_ctx = torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(amp_dtype is not None)) \
                               if (amp_dtype is not None) else nullcontext()

                for fd, ind, ctx, y in pbar:
                    fd = fd.to(CFG.device, non_blocking=HAS_CUDA)
                    ind = ind.to(CFG.device, non_blocking=HAS_CUDA)
                    ctx = ctx.to(CFG.device, non_blocking=HAS_CUDA)
                    y  = y.to(CFG.device, non_blocking=HAS_CUDA)

                    with autocast_ctx:
                        pred = model(fd, ind, ctx)
                        cc = pearsonr(pred, y)
                        loss_pearson = (1 - cc) if not torch.isnan(cc) else F.mse_loss(pred, y)
                        rank_val = pairwise_ranking_loss(pred, y, num_pairs=2048)
                        loss = w_loss * loss_pearson + (1 - w_loss) * rank_val

                    loss_scaled = loss / CFG.grad_accum_steps
                    if amp_dtype == torch.float16:
                        scaler_grad.scale(loss_scaled).backward()
                    else:
                        loss_scaled.backward()

                    do_step = ((step + 1) % CFG.grad_accum_steps == 0)
                    if do_step:
                        if amp_dtype == torch.float16:
                            scaler_grad.step(opt); scaler_grad.update()
                        else:
                            opt.step()
                        opt.zero_grad(set_to_none=True)

                    with torch.no_grad():
                        bs = y.shape[0]
                        n_sum += bs
                        loss_sum += loss.detach().float().item() * bs
                        rank_sum += rank_val.detach().float().item() * bs
                        icp = float(cc.detach().float().item()) if not torch.isnan(cc) else float("nan")
                        icr = rank_ic(pred.detach(), y.detach())
                        icp_sum += (0.0 if math.isnan(icp) else icp) * bs
                        icr_sum += icr * bs

                    step += 1
                    window_step += 1

                    if (step % max(1, getattr(CFG, "print_step_interval", 10))) == 0:
                        pbar.set_postfix(
                            loss=f"{loss_sum/max(1,n_sum):.4f}",
                            pairwise_ranking_loss=f"{rank_sum/max(1,n_sum):.4f}",
                            ic_p=f"{icp_sum/max(1,n_sum):.4f}",
                            ic_r=f"{icr_sum/max(1,n_sum):.4f}",
                            w=w_loss
                        )
                        writer.add_scalar(f"{tb_prefix}/train_step/loss", loss.detach().float().item(), window_step)
                        writer.add_scalar(f"{tb_prefix}/train_step/pairwise_ranking_loss", rank_val.detach().float().item(), window_step)
                        if not torch.isnan(cc):
                            writer.add_scalar(f"{tb_prefix}/train_step/ic_pearson", float(cc.detach().float().item()), window_step)
                        writer.add_scalar(f"{tb_prefix}/train_step/ic_rank", icr, window_step)
                        try:
                            lr_cur = opt.param_groups[0]["lr"]
                            writer.add_scalar(f"{tb_prefix}/train_step/lr", lr_cur, window_step)
                        except Exception:
                            pass
                        writer.add_scalar("global/train_step/loss", loss.detach().float().item(), global_step_step)
                        writer.add_scalar("global/train_step/pairwise_ranking_loss", rank_val.detach().float().item(), global_step_step)
                        if not torch.isnan(cc):
                            writer.add_scalar("global/train_step/ic_pearson", float(cc.detach().float().item()), global_step_step)
                        writer.add_scalar("global/train_step/ic_rank", icr, global_step_step)
                        try:
                            lr_cur = opt.param_groups[0]["lr"]
                            writer.add_scalar("global/train_step/lr", lr_cur, global_step_step)
                        except Exception:
                            pass

                    global_step_step += 1

                print(f"[{pred_date.strftime('%Y-%m-%d')}] "
                      f"epoch {epoch_id} "
                      f"Train: loss={loss_sum/max(1,n_sum):.4f} "
                      f"pairwise ranking loss={rank_sum/max(1,n_sum):.4f} "
                      f"ic_p={icp_sum/max(1,n_sum):.4f} ic_r={icr_sum/max(1,n_sum):.4f} "
                      f"w={w_loss}")

                writer.add_scalar(f"{tb_prefix}/train_epoch/loss", loss_sum/max(1,n_sum), epoch_id-1)
                writer.add_scalar(f"{tb_prefix}/train_epoch/pairwise_ranking_loss", rank_sum/max(1,n_sum), epoch_id-1)
                writer.add_scalar(f"{tb_prefix}/train_epoch/ic_pearson", icp_sum/max(1,n_sum), epoch_id-1)
                writer.add_scalar(f"{tb_prefix}/train_epoch/ic_rank", icr_sum/max(1,n_sum), epoch_id-1)

                writer.add_scalar("global/train_epoch/loss", loss_sum/max(1,n_sum), global_step_epoch)
                writer.add_scalar("global/train_epoch/pairwise_ranking_loss", rank_sum/max(1,n_sum), global_step_epoch)
                writer.add_scalar("global/train_epoch/ic_pearson", icp_sum/max(1,n_sum), global_step_epoch)
                writer.add_scalar("global/train_epoch/ic_rank", icr_sum/max(1,n_sum), global_step_epoch)

                # ---------------- 验证评估（用于早停与 best） ---------------- #
                model.eval()

                def eval_split_ram(mem_items):
                    per_date = []
                    for it in mem_items:
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
                    return per_date

                def eval_split_stream(keys):
                    per_date = []
                    for gk in keys:
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
                        per_date.append({"mse": mse, "ic_pearson": cc, "ic_rank": ric,
                                         "pairwise_rank": prl, "loss": loss_val, "n": len(y_cpu)})
                    return per_date

                per_date_val = (eval_split_ram(mem_items_val) if (use_ram and mem_items_val is not None)
                                else eval_split_stream(val_gk))
                if len(per_date_val) == 0:
                    valm = {"avg_mse": float("nan"), "avg_ic_pearson": float("nan"),
                            "avg_ic_rank": float("nan"), "std_ic_rank": float("nan"),
                            "avg_pairwise_rank": float("nan"), "avg_loss": float("nan"),
                            "dates": 0}
                    val_target = -1e9
                else:
                    n_total = sum(d["n"] for d in per_date_val)
                    wv = [d["n"] / n_total for d in per_date_val]
                    avg_mse = sum(d["mse"] * wi for d, wi in zip(per_date_val, wv))
                    avg_icp = sum((0.0 if math.isnan(d["ic_pearson"]) else d["ic_pearson"]) * wi for d, wi in zip(per_date_val, wv))
                    avg_icr = sum(d["ic_rank"] * wi for d, wi in zip(per_date_val, wv))
                    std_icr = float(np.std([d["ic_rank"] for d in per_date_val], ddof=1)) if len(per_date_val) > 1 else 0.0
                    avg_prl = sum(d["pairwise_rank"] * wi for d, wi in zip(per_date_val, wv))
                    avg_loss = sum(d["loss"] * wi for d, wi in zip(per_date_val, wv))
                    valm = {"avg_mse": float(avg_mse), "avg_ic_pearson": float(avg_icp),
                            "avg_ic_rank": float(avg_icr), "std_ic_rank": float(std_icr),
                            "avg_pairwise_rank": float(avg_prl), "avg_loss": float(avg_loss),
                            "dates": len(per_date_val)}
                    val_target = float(avg_icr)

                print(f"  Val:  mse={valm['avg_mse']:.6f} ic_p={valm['avg_ic_pearson']:.4f} "
                      f"ic_r={valm['avg_ic_rank']:.4f} prl={valm['avg_pairwise_rank']:.4f} "
                      f"loss={valm['avg_loss']:.4f} ic_r_std={valm['std_ic_rank']:.4f} dates={valm['dates']}")

                ep = epoch_id - 1
                writer.add_scalar(f"{tb_prefix}/val/mse", valm['avg_mse'], ep)
                writer.add_scalar(f"{tb_prefix}/val/ic_pearson", valm['avg_ic_pearson'], ep)
                writer.add_scalar(f"{tb_prefix}/val/ic_rank", valm['avg_ic_rank'], ep)
                writer.add_scalar(f"{tb_prefix}/val/ic_rank_std", valm['std_ic_rank'], ep)
                writer.add_scalar(f"{tb_prefix}/val/pairwise_ranking_loss", valm['avg_pairwise_rank'], ep)
                writer.add_scalar(f"{tb_prefix}/val/loss", valm['avg_loss'], ep)

                writer.add_scalar("global/val/mse", valm['avg_mse'], global_step_epoch)
                writer.add_scalar("global/val/ic_pearson", valm['avg_ic_pearson'], global_step_epoch)
                writer.add_scalar("global/val/ic_rank", valm['avg_ic_rank'], global_step_epoch)
                writer.add_scalar("global/val/ic_rank_std", valm['std_ic_rank'], global_step_epoch)
                writer.add_scalar("global/val/pairwise_ranking_loss", valm['avg_pairwise_rank'], global_step_epoch)
                writer.add_scalar("global/val/loss", valm['avg_loss'], global_step_epoch)

                # -------- 保存 best + 正确的早停判断（使用 prev_best 进行比较） --------
                prev_best = best_val_score
                if math.isfinite(val_target) and val_target > best_val_score:
                    best_val_score = val_target
                    best_epoch  = epoch_id
                    save_checkpoint(model, best_path, scaler_d=scaler_d)

                improved = (math.isfinite(val_target) and (val_target > prev_best + float(CFG.early_stop_min_delta)))
                if improved:
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                global_step_epoch += 1

                reached_min = (epoch_id >= int(CFG.early_stop_min_epochs))
                patience_out = reached_min and (epochs_no_improve >= int(CFG.early_stop_patience))
                maxed_out = (epoch_id >= int(CFG.early_stop_max_epochs))
                if patience_out or maxed_out:
                    reason = "patience" if patience_out else "max_epochs"
                    print(f"[早停] 窗口 {pred_date.strftime('%Y-%m-%d')} 在 epoch={epoch_id} 触发早停（原因：{reason}）。"
                          f" best_on_val_ic_rank={best_val_score:.4f} (best_epoch={best_epoch})")
                    break

            # 保存最后一版
            last_path = out_model_dir / f"model_{pred_date.strftime('%Y%m%d')}.pth"
            save_checkpoint(model, last_path, scaler_d=scaler_d)
            print(f"窗口结束，已保存：last -> {last_path.name} , best(ep={best_epoch}) -> {Path(best_path).name}")

            # ---------------- 写入登记表（每窗口一行） ---------------- #
            payload = torch.load(best_path, map_location=CFG.device, weights_only=False)
            state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
            model.load_state_dict(state_dict, strict=False)
            model.eval()

            def summarize(per_date):
                if len(per_date) == 0:
                    return {"avg_mse": float("nan"), "avg_ic_pearson": float("nan"),
                            "avg_ic_rank": float("nan"), "std_ic_rank": float("nan"),
                            "avg_pairwise_rank": float("nan"), "avg_loss": float("nan"),
                            "dates": 0}
                n_total = sum(d["n"] for d in per_date)
                wv = [d["n"] / n_total for d in per_date]
                avg_mse = sum(d["mse"] * wi for d, wi in zip(per_date, wv))
                avg_icp = sum((0.0 if math.isnan(d["ic_pearson"]) else d["ic_pearson"]) * wi for d, wi in zip(per_date, wv))
                avg_icr = sum(d["ic_rank"] * wi for d, wi in zip(per_date, wv))
                std_icr = float(np.std([d["ic_rank"] for d in per_date], ddof=1)) if len(per_date) > 1 else 0.0
                avg_prl = sum(d["pairwise_rank"] * wi for d, wi in zip(per_date, wv))
                avg_loss = sum(d["loss"] * wi for d, wi in zip(per_date, wv))
                return {"avg_mse": float(avg_mse), "avg_ic_pearson": float(avg_icp),
                        "avg_ic_rank": float(avg_icr), "std_ic_rank": float(std_icr),
                        "avg_pairwise_rank": float(avg_prl), "avg_loss": float(avg_loss),
                        "dates": len(per_date)}

            def eval_split_stream(keys):
                per_date = []
                for gk in keys:
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
                return per_date

            per_date_val = eval_split_stream(val_gk)
            final_val = summarize(per_date_val)

            row = {
                "pred_date": pred_date.strftime("%Y-%m-%d"),
                "best_epoch": best_epoch,
                "val_avg_pearson": final_val["avg_ic_pearson"],
                "val_avg_rankic": final_val["avg_ic_rank"],
                "val_std_rankic": final_val["std_ic_rank"],
                "val_avg_pairwise_ranking_loss": final_val["avg_pairwise_rank"],
                "val_avg_loss": final_val["avg_loss"],
                "epoch_count": epoch_id,
            }
            df_new = pd.DataFrame([row])
            registry_file.parent.mkdir(parents=True, exist_ok=True)
            if registry_file.exists():
                df_old = pd.read_csv(registry_file)
                df_all = pd.concat([df_old, df_new], ignore_index=True)
            else:
                df_all = df_new
            df_all.to_csv(registry_file, index=False)

    print("训练完成。")
    writer.close()

if __name__ == "__main__":
    main()