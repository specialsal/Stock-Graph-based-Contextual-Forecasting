# ====================== train_rolling.py ======================
# coding: utf-8
"""
滚动训练主循环（日频 + 双路行业GAT）— RankNet_margin(cost=m) 版本（精简日志）
- 仅保留 GAT 图（无 mean 分支）
- 门控嵌入：chain_sector
- 图：industry2 + industry 两路聚合并融合（hybrid，固定）
"""
import math, torch, pandas as pd, numpy as np, h5py
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from contextlib import nullcontext
from torch.utils.tensorboard import SummaryWriter
import time

from config import CFG
from model import GCFNet
from utils import weekly_fridays, load_calendar, load_industry_twolevel_map, load_chain_sector_map, rank_ic
from train_utils import (
    HAS_CUDA, setup_env, get_amp_dtype, try_fused_adamw, ensure_parent_dir,
    make_filter_fn, load_stock_info, load_flag_table,
    fit_scaler_for_groups, iterate_group_minibatches, load_group_to_memory,
    pairwise_ranking_loss_margin, pearsonr,
    load_window_to_ram, iterate_ram_minibatches
)

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

def main():
    setup_env()

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

    # 3) 三套映射与维度
    chain_map = load_chain_sector_map(CFG.style_map_file)
    ind1_map, ind2_map = load_industry_twolevel_map(CFG.industry_map_file)

    with h5py.File(daily_h5, "r") as h5d:
        d_in = len(h5d.attrs["factor_cols"])
    ctx_dim = ctx_df.shape[1]

    n_chain_known = (max(chain_map.values()) + 1) if len(chain_map)>0 else 0
    n_ind1_known  = (max(ind1_map.values()) + 1) if len(ind1_map)>0 else 0
    n_ind2_known  = (max(ind2_map.values()) + 1) if len(ind2_map)>0 else 0

    pad_chain_id = n_chain_known
    pad_ind1_id  = n_ind1_known
    pad_ind2_id  = n_ind2_known

    # 4) 模型、优化器、AMP
    amp_dtype = get_amp_dtype()
    model = GCFNet(
        d_in=d_in,
        n_chain=n_chain_known, n_ind1=n_ind1_known, n_ind2=n_ind2_known,
        ctx_dim=ctx_dim,
        hidden=CFG.hidden, chain_emb_dim=CFG.chain_emb,
        tr_layers=CFG.tr_layers, gat_layers=getattr(CFG, "gat_layers", 1)
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
            train_dates = fridays[i - train_weeks : i]
            val_dates   = fridays[i : i + val_weeks]
            pred_date   = fridays[i + val_weeks]

            train_gk = [date2gk_all[d] for d in train_dates if d in date2gk_all]
            val_gk   = [date2gk_all[d] for d in val_dates   if d in date2gk_all]

            scaler_d = fit_scaler_for_groups(h5, train_gk)

            # Warm Start（可选）
            if getattr(CFG, "warm_start_enable", False):
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
                            print(f"[WarmStart][警告] 加载上一窗口 best 失败：{e}")
                    else:
                        print(f"[WarmStart] 未找到上一窗口 best：{prev_best_path.name}（忽略）")

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
                    h5, train_gk, label_df, ctx_df, scaler_d,
                    chain_map, ind1_map, ind2_map,
                    pad_chain_id, pad_ind1_id, pad_ind2_id,
                    filter_fn=filter_fn
                )
                mem_items_val, gb_val = load_window_to_ram(
                    h5, val_gk, label_df, ctx_df, scaler_d,
                    chain_map, ind1_map, ind2_map,
                    pad_chain_id, pad_ind1_id, pad_ind2_id,
                    filter_fn=filter_fn
                )
                total_gb = gb_train + gb_val
                cap_gb = float(getattr(CFG, "ram_accel_mem_cap_gb", 48))
                if total_gb > cap_gb:
                    print(f"[RAM预载] 估算内存 {total_gb:.2f} GB 超过上限 {cap_gb:.2f} GB，回退逐组加载")
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
                prl_sum = 0.0
                icr_sum = 0.0
                n_sum = 0
                opt.zero_grad(set_to_none=True)
                step = 0

                if use_ram:
                    data_iter = iterate_ram_minibatches(mem_items_train, CFG.batch_size, shuffle=True)
                    pbar = tqdm(data_iter, total=None, leave=False, desc="train-fast[RAM]")
                else:
                    pbar = tqdm(
                        iterate_group_minibatches(
                            h5, train_gk, label_df, ctx_df, scaler_d,
                            chain_map, ind1_map, ind2_map,
                            pad_chain_id, pad_ind1_id, pad_ind2_id,
                            batch_size=CFG.batch_size, shuffle=True, filter_fn=filter_fn
                        ),
                        total=None, leave=False, desc="train-fast"
                    )

                amp_enabled = (amp_dtype is not None)
                autocast_ctx = torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled) \
                               if amp_enabled else nullcontext()

                for fd, chain, ind1, ind2, ctx, y in pbar:
                    fd = fd.to(CFG.device, non_blocking=HAS_CUDA)
                    chain = chain.to(CFG.device, non_blocking=HAS_CUDA)
                    ind1 = ind1.to(CFG.device, non_blocking=HAS_CUDA)
                    ind2 = ind2.to(CFG.device, non_blocking=HAS_CUDA)
                    ctx = ctx.to(CFG.device, non_blocking=HAS_CUDA)
                    y  = y.to(CFG.device, non_blocking=HAS_CUDA)

                    with autocast_ctx:
                        pred = model(fd, chain, ind1, ind2, ctx)
                        rank_val = pairwise_ranking_loss_margin(pred, y, m=CFG.pair_margin_m, num_pairs=CFG.pair_num_pairs)
                        loss = rank_val

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
                        prl_sum += rank_val.detach().float().item() * bs
                        icr = rank_ic(pred.detach(), y.detach())
                        icr_sum += icr * bs

                    step += 1
                    window_step += 1

                    if (step % max(1, getattr(CFG, "print_step_interval", 10))) == 0:
                        pbar.set_postfix(
                            pairwise_margin_loss=f"{prl_sum/max(1,n_sum):.4f}",
                            ic_r=f"{icr_sum/max(1,n_sum):.4f}"
                        )
                        writer.add_scalar(f"{tb_prefix}/train_step/pairwise_margin_loss", rank_val.detach().float().item(), window_step)
                        writer.add_scalar(f"{tb_prefix}/train_step/ic_rank", icr, window_step)
                        writer.add_scalar("global/train_step/pairwise_margin_loss", rank_val.detach().float().item(), global_step_step)
                        writer.add_scalar("global/train_step/ic_rank", icr, global_step_step)

                    global_step_step += 1

                print(f"[{pred_date.strftime('%Y-%m-%d')}] "
                      f"epoch {epoch_id} "
                      f"Train: pairwise_margin_loss={prl_sum/max(1,n_sum):.4f} "
                      f"ic_r={icr_sum/max(1,n_sum):.4f}")

                writer.add_scalar(f"{tb_prefix}/train_epoch/pairwise_margin_loss", prl_sum/max(1,n_sum), epoch_id-1)
                writer.add_scalar(f"{tb_prefix}/train_epoch/ic_rank", icr_sum/max(1,n_sum), epoch_id-1)
                writer.add_scalar("global/train_epoch/pairwise_margin_loss", prl_sum/max(1,n_sum), global_step_epoch)
                writer.add_scalar("global/train_epoch/ic_rank", icr_sum/max(1,n_sum), global_step_epoch)

                # ---------------- 验证 ---------------- #
                model.eval()

                def eval_split_ram(mem_items):
                    per_date = []
                    for it in mem_items:
                        X = torch.from_numpy(it["X"]).to(CFG.device, non_blocking=HAS_CUDA)
                        ch_t = torch.from_numpy(it["chain"]).to(CFG.device, non_blocking=HAS_CUDA)
                        i1_t = torch.from_numpy(it["ind1"]).to(CFG.device, non_blocking=HAS_CUDA)
                        i2_t = torch.from_numpy(it["ind2"]).to(CFG.device, non_blocking=HAS_CUDA)
                        ctx_t = torch.from_numpy(it["ctx"]).to(CFG.device, non_blocking=HAS_CUDA)
                        y_t  = torch.from_numpy(it["y"]).to(CFG.device, non_blocking=HAS_CUDA)
                        preds = []
                        bs = CFG.batch_size
                        for st in range(0, X.shape[0], bs):
                            p = model(X[st:st+bs], ch_t[st:st+bs], i1_t[st:st+bs], i2_t[st:st+bs], ctx_t[st:st+bs])
                            preds.append(p.detach().float().cpu())
                        pred_cpu = torch.cat(preds, 0)
                        y_cpu = y_t.detach().float().cpu()
                        ric  = rank_ic(pred_cpu, y_cpu)
                        prl  = float(pairwise_ranking_loss_margin(pred_cpu, y_cpu, m=CFG.pair_margin_m, num_pairs=CFG.pair_num_pairs).item())
                        per_date.append({"ic_rank": ric, "pairwise_margin_loss": prl, "n": len(y_cpu)})
                    return per_date

                def eval_split_stream(keys):
                    per_date = []
                    for gk in keys:
                        out = load_group_to_memory(h5, gk, label_df, ctx_df, scaler_d,
                                                   chain_map, ind1_map, ind2_map,
                                                   pad_chain_id, pad_ind1_id, pad_ind2_id,
                                                   filter_fn=filter_fn)
                        if out is None:
                            continue
                        X, ch_t, i1_t, i2_t, ctx_t, y_t, _ = out
                        X = torch.from_numpy(X).to(CFG.device, non_blocking=HAS_CUDA)
                        ch_t = torch.from_numpy(ch_t).to(CFG.device, non_blocking=HAS_CUDA)
                        i1_t = torch.from_numpy(i1_t).to(CFG.device, non_blocking=HAS_CUDA)
                        i2_t = torch.from_numpy(i2_t).to(CFG.device, non_blocking=HAS_CUDA)
                        ctx_t = torch.from_numpy(ctx_t).to(CFG.device, non_blocking=HAS_CUDA)
                        y_t  = torch.from_numpy(y_t).to(CFG.device, non_blocking=HAS_CUDA)
                        preds = []
                        bs = CFG.batch_size
                        for st in range(0, X.shape[0], bs):
                            p = model(X[st:st+bs], ch_t[st:st+bs], i1_t[st:st+bs], i2_t[st:st+bs], ctx_t[st:st+bs])
                            preds.append(p.detach().float().cpu())
                        pred_cpu = torch.cat(preds, 0)
                        y_cpu = y_t.detach().float().cpu()
                        ric  = rank_ic(pred_cpu, y_cpu)
                        prl  = float(pairwise_ranking_loss_margin(pred_cpu, y_cpu, m=CFG.pair_margin_m, num_pairs=CFG.pair_num_pairs).item())
                        per_date.append({"ic_rank": ric, "pairwise_margin_loss": prl, "n": len(y_cpu)})
                    return per_date

                per_date_val = (eval_split_ram(mem_items_val) if (use_ram and mem_items_val is not None)
                                else eval_split_stream(val_gk))
                if len(per_date_val) == 0:
                    val_avg_icr = float("nan")
                    val_avg_prl = float("nan")
                    val_target = -1e9
                else:
                    n_total = sum(d["n"] for d in per_date_val)
                    wv = [d["n"] / n_total for d in per_date_val]
                    val_avg_icr = sum(d["ic_rank"] * wi for d, wi in zip(per_date_val, wv))
                    val_avg_prl = sum(d["pairwise_margin_loss"] * wi for d, wi in zip(per_date_val, wv))
                    val_target = float(val_avg_icr)

                print(f"  Val:  pairwise_margin_loss={val_avg_prl:.4f} ic_r={val_avg_icr:.4f}")

                ep = epoch_id - 1
                writer.add_scalar(f"{tb_prefix}/val/pairwise_margin_loss", val_avg_prl, ep)
                writer.add_scalar(f"{tb_prefix}/val/ic_rank", val_avg_icr, ep)
                writer.add_scalar("global/val/pairwise_margin_loss", val_avg_prl, global_step_epoch)
                writer.add_scalar("global/val/ic_rank", val_avg_icr, global_step_epoch)

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
                    return {"avg_ic_rank": float("nan"),
                            "avg_pairwise_margin_loss": float("nan"),
                            "dates": 0}
                n_total = sum(d["n"] for d in per_date)
                wv = [d["n"] / n_total for d in per_date]
                avg_icr = sum(d["ic_rank"] * wi for d, wi in zip(per_date, wv))
                avg_prl = sum(d["pairwise_margin_loss"] * wi for d, wi in zip(per_date, wv))
                return {"avg_ic_rank": float(avg_icr),
                        "avg_pairwise_margin_loss": float(avg_prl),
                        "dates": len(per_date)}

            def eval_split_stream(keys):
                per_date = []
                for gk in keys:
                    out = load_group_to_memory(h5, gk, label_df, ctx_df, scaler_d,
                                               chain_map, ind1_map, ind2_map,
                                               pad_chain_id, pad_ind1_id, pad_ind2_id,
                                               filter_fn=filter_fn)
                    if out is None: continue
                    X, ch_t, i1_t, i2_t, ctx_t, y_t, _ = out
                    X = torch.from_numpy(X).to(CFG.device, non_blocking=HAS_CUDA)
                    ch_t = torch.from_numpy(ch_t).to(CFG.device, non_blocking=HAS_CUDA)
                    i1_t = torch.from_numpy(i1_t).to(CFG.device, non_blocking=HAS_CUDA)
                    i2_t = torch.from_numpy(i2_t).to(CFG.device, non_blocking=HAS_CUDA)
                    ctx_t = torch.from_numpy(ctx_t).to(CFG.device, non_blocking=HAS_CUDA)
                    y_t  = torch.from_numpy(y_t).to(CFG.device, non_blocking=HAS_CUDA)
                    preds = []
                    bs = CFG.batch_size
                    for st in range(0, X.shape[0], bs):
                        p = model(X[st:st+bs], ch_t[st:st+bs], i1_t[st:st+bs], i2_t[st:st+bs], ctx_t[st:st+bs])
                        preds.append(p.detach().float().cpu())
                    pred_cpu = torch.cat(preds, 0)
                    y_cpu = y_t.detach().float().cpu()
                    ric  = rank_ic(pred_cpu, y_cpu)
                    prl  = float(pairwise_ranking_loss_margin(pred_cpu, y_cpu, m=CFG.pair_margin_m, num_pairs=CFG.pair_num_pairs).item())
                    per_date.append({"ic_rank": ric, "pairwise_margin_loss": prl, "n": len(y_cpu)})
                return per_date

            per_date_val = eval_split_stream(val_gk)
            final_val = summarize(per_date_val)

            row = {
                "pred_date": pred_date.strftime("%Y-%m-%d"),
                "best_epoch": best_epoch,
                "val_avg_rankic": final_val["avg_ic_rank"],
                "val_avg_pairwise_margin_loss": final_val["avg_pairwise_margin_loss"],
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