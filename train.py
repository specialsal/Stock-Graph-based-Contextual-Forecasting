# coding: utf-8
"""
训练脚本：三模态 + 行业 / 板块标签
- 按“单日截面”为一个 batch 进行训练与验证
- 注意力已改为 O(B)，并加入：
  * 训练时每日样本上限（控制显存）
  * 验证时分块编码（先编码再一次性交互）
  * 混合精度（bf16/fp16）
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from contextlib import nullcontext

from config import CFG
from dataset import StockDataset, collate_fn
from model import FusionModel


# --------------- 按日期成批的 BatchSampler ----------------
class ByDateBatchSampler(Sampler):
    """
    每个 batch 是同一天（同一截面）的所有股票索引。
    - 训练：可设置 max_per_day，随机下采样到上限，并打乱日期顺序
    - 验证：不做下采样，日期升序
    """
    def __init__(self, idxs, shuffle=True, max_per_day=None):
        self.groups = {}
        for i, meta in enumerate(idxs):
            date_key = meta["date"]  # pandas.Timestamp
            if date_key not in self.groups:
                self.groups[date_key] = []
            self.groups[date_key].append(i)
        self.order = sorted(self.groups.keys())
        self.shuffle = shuffle
        self.max_per_day = max_per_day

    def __iter__(self):
        order = self.order[:]
        if self.shuffle:
            np.random.shuffle(order)
        for d in order:
            g = self.groups[d]
            if self.max_per_day is not None and len(g) > self.max_per_day:
                # 每个 epoch 重新随机子集
                yield list(np.random.choice(g, size=self.max_per_day, replace=False))
            else:
                yield g

    def __len__(self):
        return len(self.order)


# --------------- RankIC & Loss ----------------
@torch.no_grad()
def rank_ic(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """Spearman 相关系数（仅用于评估）"""
    pred_r = pred.argsort().argsort().float()
    tgt_r = tgt.argsort().argsort().float()
    pred_r = (pred_r - pred_r.mean()) / (pred_r.std() + 1e-8)
    tgt_r = (tgt_r - tgt_r.mean()) / (tgt_r.std() + 1e-8)
    return (pred_r * tgt_r).mean()


def ic_loss(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """
    可导的 IC 近似：pred 与 label 的秩（仅对 label 排名）之间的皮尔逊相关
    """
    p = (pred - pred.mean()) / (pred.std() + 1e-8)
    t_r = tgt.argsort().argsort().float()
    t = (t_r - t_r.mean()) / (t_r.std() + 1e-8)
    return 1.0 - (p * t).mean()


def pairwise_loss(pred, tgt, margin=0.01):
    """成对排序 hinge 损失"""
    n = pred.size(0)
    if n < 2:
        return torch.tensor(0.0, device=pred.device)
    idx1 = torch.randint(0, n, (n // 2,), device=pred.device)
    idx2 = torch.randint(0, n, (n // 2,), device=pred.device)
    dp = pred[idx1] - pred[idx2]
    dt = tgt[idx1] - tgt[idx2]
    label = torch.sign(dt)
    return torch.relu(margin - label * dp).mean()


def total_loss(pred, tgt):
    return CFG.alpha * ic_loss(pred, tgt) + (1 - CFG.alpha) * pairwise_loss(pred, tgt)


# --------------- 验证：分块编码 + 一次性交互 ----------------
@torch.no_grad()
def forward_day_with_chunked_encode(model, batch, amp_ctx, amp_kwargs):
    """
    为了降低显存，验证时将 [B, Td/Tm] 的编码部分分块进行，
    得到整日的 stk_repr 后，再一次性交互与预测。
    """
    B = batch["daily"].size(0)
    device = batch["daily"].device
    chunk = CFG.val_encode_chunk if CFG.val_encode_chunk and CFG.val_encode_chunk > 0 else B

    repr_list = []
    for s in range(0, B, chunk):
        e = min(B, s + chunk)
        with amp_ctx(**amp_kwargs):
            stk_repr_chunk = model.encode(
                batch["daily"][s:e],
                batch["min30"][s:e],
                batch["ind_id"][s:e],
                batch["sec_id"][s:e]
            )  # [e-s, D]
        repr_list.append(stk_repr_chunk)
    stk_repr = torch.cat(repr_list, dim=0)  # [B, D]

    # 交互部分一次性完成（O(B)注意力）
    with amp_ctx(**amp_kwargs):
        pred = model.interact_and_head(stk_repr)  # [B]
    return pred


# --------------- 训练 / 验证 ----------------
@torch.no_grad()
def evaluate(model, loader, amp_ctx, amp_kwargs):
    model.eval()
    tot_ic, tot_loss, tot_ic_loss, tot_pair_loss, n = 0.0, 0.0, 0.0, 0.0, 0

    pbar = tqdm(loader, desc="Validating", leave=False)
    for batch in pbar:
        # 确保 batch 中只有一个日期（单日截面）
        assert batch["date"].unique().numel() == 1, "验证集 batch 混入了多个日期，请检查采样器"

        for k in ("daily", "min30", "ind_id", "sec_id", "label"):
            batch[k] = batch[k].to(CFG.device, non_blocking=True)

        pred = forward_day_with_chunked_encode(model, batch, amp_ctx, amp_kwargs)

        # 计算各种损失
        ic_loss_val = ic_loss(pred, batch["label"])
        pair_loss_val = pairwise_loss(pred, batch["label"])
        loss = CFG.alpha * ic_loss_val + (1 - CFG.alpha) * pair_loss_val
        ic = rank_ic(pred, batch["label"]).item()

        tot_ic += ic
        tot_loss += loss.item()
        tot_ic_loss += ic_loss_val.item()
        tot_pair_loss += pair_loss_val.item()
        n += 1

        pbar.set_postfix({
            'Val_Loss': f'{tot_loss / n:.4f}',
            'Val_IC': f'{tot_ic / n:.4f}',
            'Val_IC_Loss': f'{tot_ic_loss / n:.4f}',
            'Val_Pair_Loss': f'{tot_pair_loss / n:.4f}'
        })

    return tot_ic / n, tot_loss / n, tot_ic_loss / n, tot_pair_loss / n


def train_one_epoch(model, loader, optimizer, scaler, amp_ctx, amp_kwargs, epoch):
    model.train()
    tot_loss, tot_ic, tot_ic_loss, tot_pair_loss, n = 0.0, 0.0, 0.0, 0.0, 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        # 确保 batch 中只有一个日期（单日截面）
        assert batch["date"].unique().numel() == 1, "训练集 batch 混入了多个日期，请检查采样器"

        for k in ("daily", "min30", "ind_id", "sec_id", "label"):
            batch[k] = batch[k].to(CFG.device, non_blocking=True)

        with amp_ctx(**amp_kwargs):
            pred = model(batch["daily"], batch["min30"], batch["ind_id"], batch["sec_id"])

            ic_loss_val = ic_loss(pred, batch["label"])
            pair_loss_val = pairwise_loss(pred, batch["label"])
            loss = CFG.alpha * ic_loss_val + (1 - CFG.alpha) * pair_loss_val

        optimizer.zero_grad(set_to_none=True)

        # AMP 反向 + 梯度裁剪
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # 计算指标
        ic = rank_ic(pred, batch["label"]).item()
        tot_loss += loss.item()
        tot_ic += ic
        tot_ic_loss += ic_loss_val.item()
        tot_pair_loss += pair_loss_val.item()
        n += 1

        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'Loss': f'{tot_loss / n:.4f}',
            'IC': f'{tot_ic / n:.4f}',
            'IC_Loss': f'{tot_ic_loss / n:.4f}',
            'Pair_Loss': f'{tot_pair_loss / n:.4f}',
            'LR': f'{current_lr:.2e}',
            'Batch': f'{batch_idx + 1}/{len(loader)}'
        })

    return tot_loss / n, tot_ic / n, tot_ic_loss / n, tot_pair_loss / n


# --------------- 主函数 ----------------
def main():
    # ---------- AMP 上下文 ----------
    use_amp = CFG.use_amp and (CFG.device.type == 'cuda')
    if use_amp:
        dtype = torch.bfloat16 if CFG.amp_dtype.lower() == "bf16" else torch.float16
        amp_ctx = torch.cuda.amp.autocast
        amp_kwargs = dict(dtype=dtype)
        scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))
    else:
        amp_ctx = nullcontext
        amp_kwargs = {}
        scaler = None

    # ---------- 数据集 ----------
    print("正在加载数据集...")
    train_ds = StockDataset(CFG.train_features_file, CFG.label_file,
                            CFG.scaler_file, CFG.universe_file)
    val_ds = StockDataset(CFG.val_features_file, CFG.label_file,
                          CFG.scaler_file, CFG.universe_file)

    # 按日成批采样器
    train_bs = ByDateBatchSampler(train_ds.idxs, shuffle=True,
                                  max_per_day=CFG.max_stocks_per_day_train)
    val_bs = ByDateBatchSampler(val_ds.idxs, shuffle=False, max_per_day=None)

    # 使用 batch_sampler，不再传 batch_size / shuffle
    train_ld = DataLoader(train_ds, batch_sampler=train_bs,
                          num_workers=CFG.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_ld = DataLoader(val_ds, batch_sampler=val_bs,
                        num_workers=CFG.num_workers, collate_fn=collate_fn, pin_memory=True)

    print(f"训练集样本数: {len(train_ds)}")
    print(f"验证集样本数: {len(val_ds)}")
    print(f"每epoch训练步数(按日): {len(train_ld)}")

    # ---------- 模型 ----------
    model = FusionModel(num_industries=train_ds.num_industries,
                        num_sectors=train_ds.num_sectors).to(CFG.device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params / 1e6:.2f} M")
    print(f"可训练参数量: {trainable_params / 1e6:.2f} M")

    # ---------- 优化器 & 调度 ----------
    opt = AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    sch = CosineAnnealingLR(opt, T_max=CFG.epochs, eta_min=CFG.lr * 0.01)

    best_ic = -1.0
    best_epoch = 0

    print(f"\n开始训练 - 设备: {CFG.device}")
    print(f"配置: alpha={CFG.alpha}, lr={CFG.lr}, weight_decay={CFG.weight_decay}, epochs={CFG.epochs}")
    print(f"AMP: {use_amp}, dtype: {CFG.amp_dtype}")
    print("=" * 80)

    for epoch in range(1, CFG.epochs + 1):
        print(f"\nEpoch {epoch}/{CFG.epochs}")
        print("-" * 50)

        # 训练
        tr_loss, tr_ic, tr_ic_loss, tr_pair_loss = train_one_epoch(model, train_ld, opt, scaler, amp_ctx, amp_kwargs, epoch)

        # 验证
        val_ic, val_loss, val_ic_loss, val_pair_loss = evaluate(model, val_ld, amp_ctx, amp_kwargs)

        # 学习率调度
        sch.step()
        current_lr = opt.param_groups[0]['lr']

        # 打印epoch总结
        print(
            f"Train - Loss: {tr_loss:.6f} | IC: {tr_ic:.6f} | IC_Loss: {tr_ic_loss:.6f} | Pair_Loss: {tr_pair_loss:.6f}")
        print(
            f"Valid - Loss: {val_loss:.6f} | IC: {val_ic:.6f} | IC_Loss: {val_ic_loss:.6f} | Pair_Loss: {val_pair_loss:.6f}")
        print(f"LR: {current_lr:.2e}")

        # 保存最好模型
        if val_ic > best_ic:
            best_ic = val_ic
            best_epoch = epoch
            torch.save({
                "model": model.state_dict(),
                "ic": best_ic,
                "epoch": epoch,
                "train_ic": tr_ic,
                "train_loss": tr_loss,
                "train_ic_loss": tr_ic_loss,
                "train_pair_loss": tr_pair_loss,
                "val_loss": val_loss,
                "val_ic_loss": val_ic_loss,
                "val_pair_loss": val_pair_loss,
                "config": CFG.__dict__.copy() if hasattr(CFG, '__dict__') else None
            }, CFG.processed_dir / "best_model.pth")
            print(f">>> 保存新最佳模型 (IC: {best_ic:.4f})")
        else:
            print(f">>> 当前最佳IC: {best_ic:.4f} (Epoch {best_epoch})")

    print("\n" + "=" * 80)
    print(f"训练结束！")
    print(f"最佳验证 IC: {best_ic:.4f} (Epoch {best_epoch})")
    print(f"模型已保存至: {CFG.processed_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()