import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
from utils import calculate_metrics


class ExperimentEngine:
    def __init__(self, model, train_loader, test_loader, device, save_dir, logger):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader  # batch_size 建议为 1
        self.device = device
        self.save_dir = save_dir
        self.logger = logger
        self.criterion = nn.CrossEntropyLoss()

    def train(self, epochs=150, lr=0.001):
        self.logger.info(f"START TRAINING for {epochs} epochs...")
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.0005)

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            # 使用tqdm显示进度条
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device).long().squeeze(1)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})

            avg_loss = epoch_loss / len(self.train_loader)
            # 每10个epoch打印一次日志，避免刷屏
            if (epoch + 1) % 10 == 0:
                self.logger.info(f"Epoch [{epoch + 1}/{epochs}], Avg Loss: {avg_loss:.4f}")

        # 保存最终模型
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'final_model.pth'))
        self.logger.info("Training Finished.")

    def evaluate(self):
        """执行 PreCM 论文中的全套评估：0, 90, 180, 270, Random"""
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
        total_iou, total_miou, total_dice, total_rd = 0, 0, 0, 0
        count = 0

        with torch.no_grad():
            for images, masks in self.test_loader:
                images = images.to(self.device)
                masks = masks.to(self.device).squeeze(1)
                bs = images.size(0)

                # 1. 确定旋转角度
                if angle_mode == 'random':
                    angles = [random.uniform(0, 360) for _ in range(bs)]
                else:
                    angles = [float(angle_mode)] * bs

                # 2. 旋转输入图像
                rot_images = torch.stack([TF.rotate(img, ang) for img, ang in zip(images, angles)])

                # 3. 推理旋转后的图像 -> 得到 Rot_Pred
                out_rot = self.model(rot_images)
                pred_rot = torch.argmax(out_rot, dim=1).float()

                # 4. 推理原始图像 -> 得到 Origin_Pred (用于计算 RD)
                out_origin = self.model(images)
                pred_origin = torch.argmax(out_origin, dim=1).float()

                # 5. 将 Rot_Pred 逆旋转回原始方向 -> 得到 Pred_Back
                pred_back = torch.stack([TF.rotate(p.unsqueeze(0), -ang).squeeze(0)
                                         for p, ang in zip(pred_rot, angles)])

                # 6. 计算指标
                # 分割指标：Pred_Back vs GT Mask
                iou, miou, dice = calculate_metrics(pred_back, masks)

                # 等变性指标 RD: |Pred_Back - Pred_Origin|
                # PreCM定义：差异像素占比
                diff = torch.abs(pred_back - pred_origin)
                rd = diff.mean().item() * 100

                total_iou += iou;
                total_miou += miou;
                total_dice += dice;
                total_rd += rd
                count += 1

        return {
            'iou': total_iou / count, 'miou': total_miou / count,
            'dice': total_dice / count, 'rd': total_rd / count
        }

    def visualize(self, num_samples=5):
        """
        完整可视化：保存 Input, GT, Pred(0), Pred(90), Diff Map
        随机抽取 num_samples 张测试图进行可视化
        """
        self.logger.info(f"START VISUALIZATION (Saving {num_samples} samples)...")
        self.model.eval()
        vis_dir = os.path.join(self.save_dir, 'vis_results')
        os.makedirs(vis_dir, exist_ok=True)

        # 随机采样
        indices = random.sample(range(len(self.test_loader.dataset)), num_samples)

        with torch.no_grad():
            for i, idx in enumerate(indices):
                image, mask = self.test_loader.dataset[idx]
                image = image.unsqueeze(0).to(self.device)  # (1, 3, H, W)

                # 1. 原始预测
                out_0 = self.model(image)
                pred_0 = torch.argmax(out_0, dim=1).float()

                # 2. 旋转90度预测
                img_90 = TF.rotate(image, 90)
                out_90 = self.model(img_90)
                pred_90_rot = torch.argmax(out_90, dim=1).float()
                pred_90_back = TF.rotate(pred_90_rot, -90)  # 转回来

                # 3. 计算差异图
                diff_map = torch.abs(pred_0 - pred_90_back).cpu().squeeze()

                # 4. 绘图
                fig, axs = plt.subplots(1, 5, figsize=(15, 3))
                # Input
                axs[0].imshow(image.cpu().squeeze().permute(1, 2, 0))
                axs[0].set_title("Input")
                axs[0].axis('off')
                # GT
                axs[1].imshow(mask.squeeze(), cmap='gray')
                axs[1].set_title("GT")
                axs[1].axis('off')
                # Pred 0
                axs[2].imshow(pred_0.cpu().squeeze(), cmap='gray')
                axs[2].set_title("Pred 0°")
                axs[2].axis('off')
                # Pred 90 (Back)
                axs[3].imshow(pred_90_back.cpu().squeeze(), cmap='gray')
                axs[3].set_title("Pred 90°(Back)")
                axs[3].axis('off')
                # Diff Map
                im = axs[4].imshow(diff_map, cmap='hot', vmin=0, vmax=1)
                axs[4].set_title("Diff (0° vs 90°)")
                axs[4].axis('off')

                plt.colorbar(im, ax=axs[4], fraction=0.046, pad=0.04)
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, f'sample_{i}.png'))
                plt.close()

        self.logger.info(f"Visualization saved to {vis_dir}")