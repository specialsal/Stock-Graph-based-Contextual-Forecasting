# coding: utf-8
"""
训练主程序
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
from pathlib import Path

from config import CFG
from dataset import StockDataset, collate_fn
from model import FusionModel


def rank_ic(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """计算Rank IC (Spearman相关系数)"""
    pred_rank = pred.argsort().argsort().float()
    target_rank = target.argsort().argsort().float()

    pred_rank = (pred_rank - pred_rank.mean()) / (pred_rank.std() + 1e-8)
    target_rank = (target_rank - target_rank.mean()) / (target_rank.std() + 1e-8)

    ic = (pred_rank * target_rank).mean()
    return ic


def ic_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """IC Loss"""
    return 1 - rank_ic(pred, target)


def pairwise_ranking_loss(pred: torch.Tensor, target: torch.Tensor, margin: float = 0.01) -> torch.Tensor:
    """成对排序损失"""
    n = pred.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=pred.device)

    # 随机采样配对
    idx1 = torch.randint(0, n, (n // 2,))
    idx2 = torch.randint(0, n, (n // 2,))

    diff_pred = pred[idx1] - pred[idx2]
    diff_true = target[idx1] - target[idx2]

    label = torch.sign(diff_true)
    loss = torch.relu(margin - label * diff_pred)

    return loss.mean()


def combined_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.7) -> torch.Tensor:
    """组合损失函数"""
    l_ic = ic_loss(pred, target)
    l_rank = pairwise_ranking_loss(pred, target)
    return alpha * l_ic + (1 - alpha) * l_rank


def train_epoch(model, loader, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_ic = 0
    num_batches = 0

    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        daily = batch['daily'].to(device)
        min30 = batch['min30'].to(device)
        labels = batch['label'].to(device)

        # 前向传播
        pred = model(daily, min30)

        # 计算损失
        loss = combined_loss(pred, labels, alpha=CFG.alpha)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 记录指标
        with torch.no_grad():
            batch_ic = rank_ic(pred, labels)
            total_loss += loss.item()
            total_ic += batch_ic.item()
            num_batches += 1

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ic': f'{batch_ic.item():.4f}'
        })

    return total_loss / num_batches, total_ic / num_batches


@torch.no_grad()
def evaluate(model, loader, device):
    """评估模型"""
    model.eval()
    total_ic = 0
    num_batches = 0

    all_preds = []
    all_labels = []

    for batch in tqdm(loader, desc="Evaluating"):
        daily = batch['daily'].to(device)
        min30 = batch['min30'].to(device)
        labels = batch['label'].to(device)

        pred = model(daily, min30)

        batch_ic = rank_ic(pred, labels)
        total_ic += batch_ic.item()
        num_batches += 1

        all_preds.append(pred.cpu())
        all_labels.append(labels.cpu())

    # 计算整体IC
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    overall_ic = rank_ic(all_preds, all_labels)

    return total_ic / num_batches, overall_ic.item()


def main():
    """主训练函数"""

    # 创建数据集
    print("加载数据集...")
    train_dataset = StockDataset(
        features_path=CFG.train_features_file,
        label_path=CFG.label_file,
        scaler_path=CFG.scaler_file,
        universe_path=CFG.universe_file
    )

    val_dataset = StockDataset(
        features_path=CFG.val_features_file,
        label_path=CFG.label_file,
        scaler_path=CFG.scaler_file,
        universe_path=CFG.universe_file
    )

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # 创建模型
    model = FusionModel().to(CFG.device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 优化器和调度器
    optimizer = AdamW(
        model.parameters(),
        lr=CFG.lr,
        weight_decay=CFG.weight_decay
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=CFG.epochs,
        eta_min=CFG.lr * 0.01
    )

    # 训练循环
    best_val_ic = -1
    best_epoch = 0

    for epoch in range(1, CFG.epochs + 1):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch}/{CFG.epochs}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # 训练
        train_loss, train_ic = train_epoch(model, train_loader, optimizer, CFG.device)

        # 验证
        val_ic, val_overall_ic = evaluate(model, val_loader, CFG.device)

        # 学习率调度
        scheduler.step()

        # 打印结果
        print(f"Train Loss: {train_loss:.4f}, Train IC: {train_ic:.4f}")
        print(f"Val IC: {val_ic:.4f}, Val Overall IC: {val_overall_ic:.4f}")

        # 保存最佳模型
        if val_overall_ic > best_val_ic:
            best_val_ic = val_overall_ic
            best_epoch = epoch

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_ic': val_overall_ic,
            }, CFG.processed_dir / 'best_model.pth')

            print(f"保存最佳模型 (IC: {val_overall_ic:.4f})")

    print(f"\n训练完成！最佳验证IC: {best_val_ic:.4f} (Epoch {best_epoch})")

    # 测试集评估
    if CFG.test_features_file.exists():
        print("\n在测试集上评估...")
        test_dataset = StockDataset(
            features_path=CFG.test_features_file,
            label_path=CFG.label_file,
            scaler_path=CFG.scaler_file,
            universe_path=CFG.universe_file
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=CFG.batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
            collate_fn=collate_fn
        )

        # 加载最佳模型
        checkpoint = torch.load(CFG.processed_dir / 'best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])

        test_ic, test_overall_ic = evaluate(model, test_loader, CFG.device)
        print(f"测试集 IC: {test_ic:.4f}, Overall IC: {test_overall_ic:.4f}")


if __name__ == "__main__":
    main()