# engine.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
import numpy as np
import random
from tqdm import tqdm
from utils import calculate_metrics


# === 新增：Dice Loss ===
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # pred: (B, 2, H, W) -> Softmax -> 取 class 1 的概率
        # target: (B, H, W) -> 0/1

        # 注意：PreCM 的输出是 (B, 2, H, W)，我们需要取第二个通道(水体)的概率
        probs = torch.softmax(pred, dim=1)[:, 1, :, :]  # (B, H, W)

        flat_pred = probs.contiguous().view(-1)
        flat_target = target.contiguous().view(-1).float()

        intersection = (flat_pred * flat_target).sum()
        dice = (2. * intersection + self.smooth) / (flat_pred.sum() + flat_target.sum() + self.smooth)

        return 1 - dice


class ExperimentEngine:
    def __init__(self, model, train_loader, test_loader, device, save_dir, logger):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.save_dir = save_dir
        self.logger = logger

        # === 修改 1: 混合 Loss (CE + Dice) ===
        self.ce_criterion = nn.CrossEntropyLoss()
        self.dice_criterion = DiceLoss()

    def train(self, epochs=150, lr=0.001):
        self.logger.info(f"START TRAINING for {epochs} epochs...")

        # 优化器
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.0005)

        # === 修改 2: 学习率调度器 (Cosine Annealing) ===
        # 让 LR 从 0.001 慢慢降到 0，帮助模型收敛到更优解
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        best_miou = 0.0

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device).long().squeeze(1)

                optimizer.zero_grad()
                outputs = self.model(images)  # (B, 2, H, W)

                # 计算混合 Loss
                loss_ce = self.ce_criterion(outputs, masks)
                loss_dice = self.dice_criterion(outputs, masks)
                loss = 0.5 * loss_ce + 0.5 * loss_dice  # 权重可以调，通常 1:1 稳健

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix({'loss': loss.item(), 'lr': optimizer.param_groups[0]['lr']})

            # 更新学习率
            scheduler.step()

            avg_loss = epoch_loss / len(self.train_loader)

            # 日志
            if (epoch + 1) % 10 == 0:
                self.logger.info(
                    f"Epoch [{epoch + 1}/{epochs}], Avg Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        # 训练结束保存模型
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'final_model.pth'))
        self.logger.info("Training Finished.")

    def evaluate(self):
        """PreCM 论文评估逻辑 (保持不变，逻辑是没问题的)"""
        self.logger.info("START EVALUATION...")
        self.model.eval()

        test_modes = ['0', '90', '180', '270', 'random']
        results = {}

        for mode in test_modes:
            metrics = self._evaluate_single_mode(mode)
            results[mode] = metrics
            self.logger.info(
                f"[{mode}] IOU: {metrics['iou']:.2f} | mIOU: {metrics['miou']:.2f} | DICE: {metrics['dice']:.2f} | RD: {metrics['rd']:.2f}")

        return results

    def _evaluate_single_mode(self, angle_mode):
        # ... (这部分代码无需修改，逻辑正确) ...
        # (为了节省篇幅，这里复用你原来的代码，只需要注意 TF.rotate 和 calculate_metrics 的配合)
        total_iou, total_miou, total_dice, total_rd = 0, 0, 0, 0
        count = 0

        with torch.no_grad():
            for images, masks in self.test_loader:
                images = images.to(self.device)
                masks = masks.to(self.device).squeeze(1)  # GT: (B, H, W)
                bs = images.size(0)

                if angle_mode == 'random':
                    angles = [random.uniform(0, 360) for _ in range(bs)]
                else:
                    angles = [float(angle_mode)] * bs

                # 旋转输入
                rot_images = torch.stack([TF.rotate(img, ang) for img, ang in zip(images, angles)])

                # 预测
                out_rot = self.model(rot_images)
                pred_rot = torch.argmax(out_rot, dim=1).float()  # (B, H, W)

                # 原始预测 (用于 RD)
                out_origin = self.model(images)
                pred_origin = torch.argmax(out_origin, dim=1).float()

                # 逆旋转
                pred_back = torch.stack([TF.rotate(p.unsqueeze(0), -ang).squeeze(0)
                                         for p, ang in zip(pred_rot, angles)])

                # 计算指标
                iou, miou, dice = calculate_metrics(pred_back, masks)

                # RD: 计算不一致的像素比例
                # diff 是 0/1 矩阵，mean() 就是比例
                diff = torch.abs(pred_back - pred_origin)
                rd = diff.mean().item() * 100

                total_iou += iou
                total_miou += miou
                total_dice += dice
                total_rd += rd
                count += 1

        return {
            'iou': total_iou / count, 'miou': total_miou / count,
            'dice': total_dice / count, 'rd': total_rd / count
        }