# coding: utf-8
"""
训练脚本（改进版）
"""
import math, numpy as np, torch
from torch.utils.data import DataLoader, Sampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
from contextlib import nullcontext

from config import CFG
from dataset import StockDataset, collate_fn
from model import FusionModel
from utils import soft_ic_loss

# ---------------- 按日期 BatchSampler ----------------
class ByDateBatchSampler(Sampler):
    def __init__(self, idxs, shuffle=True, max_per_day=None):
        self.groups = {}
        for i, meta in enumerate(idxs):
            date_key = meta["date"]
            self.groups.setdefault(date_key, []).append(i)
        self.order = sorted(self.groups.keys())
        self.shuffle, self.max_per_day = shuffle, max_per_day

    def __iter__(self):
        order = self.order[:]
        if self.shuffle:
            np.random.shuffle(order)
        for d in order:
            g = self.groups[d]
            if self.max_per_day and len(g) > self.max_per_day:
                yield list(np.random.choice(g, self.max_per_day, replace=False))
            else:
                yield g

    def __len__(self):
        return len(self.order)


# ---------------- Pairwise Loss ----------------
def pairwise_loss(pred: torch.Tensor,
                  tgt:  torch.Tensor,
                  margin: float = 0.05) -> torch.Tensor:
    """
    正负对：label 高于中位数视为正，其余为负
    """
    pos_mask = tgt > tgt.median()
    neg_mask = ~pos_mask
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return torch.tensor(0., device=pred.device)

    pos_preds = pred[pos_mask]
    neg_preds = pred[neg_mask]

    # 随机对齐长度
    n = min(len(pos_preds), len(neg_preds))
    pos_preds = pos_preds[torch.randperm(len(pos_preds))[:n]]
    neg_preds = neg_preds[torch.randperm(len(neg_preds))[:n]]

    loss = torch.relu(margin - (pos_preds - neg_preds)).mean()
    return loss


# ---------------- RankIC ----------------
@torch.no_grad()
def rank_ic(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    pr = pred.argsort().argsort().float()
    tr = tgt.argsort().argsort().float()
    pr = (pr - pr.mean()) / (pr.std() + 1e-8)
    tr = (tr - tr.mean()) / (tr.std() + 1e-8)
    return (pr * tr).mean()


# ---------------- Evaluate ----------------
@torch.no_grad()
def evaluate(model, loader, amp_ctx, amp_kwargs):
    model.eval()
    tot_ic_w, tot_loss, tot_pair, n_stk = 0., 0., 0., 0

    pbar = tqdm(loader, desc="Valid", leave=False)
    for batch in pbar:
        for k in ("daily", "min30", "ind_id", "sec_id", "label"):
            batch[k] = batch[k].to(CFG.device, non_blocking=True)

        with amp_ctx(**amp_kwargs):
            pred = model(batch["daily"], batch["min30"],
                         batch["ind_id"], batch["sec_id"])
        ic_l   = soft_ic_loss(pred, batch["label"])
        pair_l = pairwise_loss(pred, batch["label"])
        loss   = CFG.alpha * ic_l + (1 - CFG.alpha) * pair_l

        B = batch["label"].size(0)
        ic = rank_ic(pred, batch["label"]).item()

        tot_ic_w += ic * B
        tot_loss += loss.item() * B
        tot_pair += pair_l.item() * B
        n_stk    += B

        pbar.set_postfix(IC=f"{tot_ic_w / n_stk:.4f}",
                         Loss=f"{tot_loss / n_stk:.4f}",
                         PairLoss=f"{tot_pair / n_stk:.4f}")

    return tot_ic_w / n_stk, tot_loss / n_stk, tot_pair / n_stk


# ---------------- Train one epoch ----------------
def train_one_epoch(model, loader, optim, scaler,
                    amp_ctx, amp_kwargs, epoch):
    model.train()
    tot_ic_w, tot_loss, tot_pair, n_stk = 0., 0., 0., 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        for k in ("daily", "min30", "ind_id", "sec_id", "label"):
            batch[k] = batch[k].to(CFG.device, non_blocking=True)

        with amp_ctx(**amp_kwargs):
            pred = model(batch["daily"], batch["min30"],
                         batch["ind_id"], batch["sec_id"])
            ic_l   = soft_ic_loss(pred, batch["label"])
            pair_l = pairwise_loss(pred, batch["label"])
            loss   = CFG.alpha * ic_l + (1 - CFG.alpha) * pair_l

        optim.zero_grad(set_to_none=True)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

        B = batch["label"].size(0)
        ic = rank_ic(pred, batch["label"]).item()

        tot_ic_w += ic * B
        tot_loss += loss.item() * B
        tot_pair += pair_l.item() * B
        n_stk    += B

        pbar.set_postfix(IC=f"{tot_ic_w / n_stk:.4f}",
                         Loss=f"{tot_loss / n_stk:.4f}",
                         PairLoss=f"{tot_pair / n_stk:.4f}",
                         LR=f"{optim.param_groups[0]['lr']:.2e}")

    return tot_ic_w / n_stk, tot_loss / n_stk, tot_pair / n_stk


# ---------------- Main ----------------
def main():
    use_amp = CFG.use_amp and CFG.device.type == "cuda"
    if use_amp:
        dtype = torch.bfloat16 if CFG.amp_dtype.lower() == "bf16" else torch.float16
        amp_ctx, amp_kwargs = torch.amp.autocast, dict(device_type='cuda', dtype=dtype)
        # GradScaler 维持原写法即可（无警告）
        scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))
    else:
        amp_ctx, amp_kwargs, scaler = nullcontext, {}, None

    # Dataset / Loader
    print("加载数据集...")
    tr_ds = StockDataset(CFG.train_features_file, CFG.label_file,
                         CFG.scaler_file, CFG.universe_file)
    val_ds = StockDataset(CFG.val_features_file, CFG.label_file,
                          CFG.scaler_file, CFG.universe_file)

    tr_bs = ByDateBatchSampler(tr_ds.idxs, shuffle=True,
                               max_per_day=CFG.max_stocks_per_day_train)
    val_bs = ByDateBatchSampler(val_ds.idxs, shuffle=False)

    tr_ld = DataLoader(tr_ds, batch_sampler=tr_bs,
                       num_workers=CFG.num_workers, collate_fn=collate_fn,
                       pin_memory=True)
    val_ld = DataLoader(val_ds, batch_sampler=val_bs,
                        num_workers=CFG.num_workers, collate_fn=collate_fn,
                        pin_memory=True)

    # Model
    model = FusionModel(tr_ds.num_industries,
                        tr_ds.num_sectors).to(CFG.device)
    print(f"参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    # Optim & Scheduler
    optim = AdamW(model.parameters(), lr=CFG.lr,
                  weight_decay=CFG.weight_decay)
    sched = CosineAnnealingWarmRestarts(optim, T_0=50, T_mult=2,
                                        eta_min=CFG.lr * 0.01)

    best_ic, best_ep = -1., 0
    print("\n开始训练...")
    for ep in range(1, CFG.epochs + 1):
        tr_ic, tr_loss, tr_pair = train_one_epoch(model, tr_ld, optim, scaler,
                                                  amp_ctx, amp_kwargs, ep)
        val_ic, val_loss, val_pair = evaluate(model, val_ld, amp_ctx, amp_kwargs)

        sched.step()

        print(f"[Epoch {ep}] "
              f"TrainIC: {tr_ic:.4f}  ValIC: {val_ic:.4f}  "
              f"TrainLoss: {tr_loss:.4f}  ValLoss: {val_loss:.4f}  "
              f"TrainPairLoss: {tr_pair:.4f}  ValPairLoss: {val_pair:.4f}")

        if val_ic > best_ic:
            best_ic, best_ep = val_ic, ep
            torch.save({"model": model.state_dict(),
                        "ic": best_ic,
                        "epoch": ep}, CFG.processed_dir / "best_model.pth")
            print(f">>> 保存最佳模型 @ Epoch {ep} (IC={best_ic:.4f})")

    print(f"\n训练结束！最优验证 IC={best_ic:.4f} (Epoch {best_ep})")


if __name__ == "__main__":
    main()