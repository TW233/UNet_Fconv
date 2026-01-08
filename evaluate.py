import torch
import numpy as np
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from dataset import WaterDataset
from networks import unet_fconv
import torch.nn.functional as F
import random


def calculate_metrics(pred, target):
    # pred: (N, H, W), target: (N, H, W)
    # 二分类：背景0，水体1
    smooth = 1e-5
    tp = ((pred == 1) & (target == 1)).sum().item()
    fp = ((pred == 1) & (target == 0)).sum().item()
    fn = ((pred == 0) & (target == 1)).sum().item()
    tn = ((pred == 0) & (target == 0)).sum().item()

    iou = tp / (tp + fp + fn + smooth)
    dice = 2 * tp / (2 * tp + fp + fn + smooth)

    # mIOU 是两类的平均 IOU
    iou_bg = tn / (tn + fp + fn + smooth)
    miou = (iou + iou_bg) / 2

    return iou * 100, miou * 100, dice * 100


def get_rotation_matrix(angle, center, scale=1.0):
    # 辅助函数，如果需要非90度倍数旋转时用，但这里主要用 TF.rotate
    pass


def evaluate_rotation(model, loader, angle_mode='0', device='cuda'):
    model.eval()
    total_iou, total_miou, total_dice, total_rd = 0, 0, 0, 0
    count = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            # masks 不需要转，因为我们算指标是把预测结果转回来和原始mask比，或者都在旋转域比
            # 但 PreCM 的 RD 是衡量“等变性误差”。
            # 这里我们处理逻辑：
            # 1. 旋转输入 Image -> Rot_Image
            # 2. 预测 Rot_Image -> Pred_Rot
            # 3. 将 Pred_Rot 反向旋转回原角度 -> Pred_Back
            # 4. 用 Pred_Back 和 原始 Mask 算 IOU/Dice (这是测试鲁棒性)
            # 5. 用 Pred_Back 和 原始输入产生的 Pred_Origin 算 RD (这是算等变误差)

            # 生成旋转角度
            batch_size = images.size(0)
            if angle_mode == 'random':
                angles = [random.uniform(0, 360) for _ in range(batch_size)]
            else:
                angles = [float(angle_mode)] * batch_size

            # 旋转输入
            rot_images = torch.stack([TF.rotate(img, angle) for img, angle in zip(images, angles)])

            # 模型预测
            outputs_rot = model(rot_images)  # (N, 2, H, W)
            preds_rot = torch.argmax(outputs_rot, dim=1).float()  # (N, H, W)

            # 获取原始预测（用于计算 RD）
            outputs_origin = model(images)
            preds_origin = torch.argmax(outputs_origin, dim=1).float()

            # 将旋转后的预测转回来
            preds_back = torch.stack([TF.rotate(pred.unsqueeze(0), -angle).squeeze(0)
                                      for pred, angle in zip(preds_rot, angles)])

            # 计算 RD: Rotation Difference
            # RD = |Pred_Back - Pred_Origin| / Pixel_Count
            # PreCM 论文公式(25)的意思是：旋转后的输出转回来，应该和原图的输出一样
            diff = torch.abs(preds_back - preds_origin)
            rd = diff.mean().item() * 100  # 转为百分比

            # 计算分割指标 (与 Ground Truth 对比)
            # 注意：masks 在 GPU 上
            masks = masks.to(device).squeeze(1)
            iou, miou, dice = calculate_metrics(preds_back, masks)

            total_iou += iou
            total_miou += miou
            total_dice += dice
            total_rd += rd
            count += 1

    print(
        f"Angle: {angle_mode} | IOU: {total_iou / count:.2f} | mIOU: {total_miou / count:.2f} | DICE: {total_dice / count:.2f} | RD: {total_rd / count:.2f}")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = unet_fconv(in_channels=3, classes=2).to(device)
    model.load_state_dict(torch.load('checkpoints/fconv_unet_epoch_150.pth'))

    test_dataset = WaterDataset(root_dir='./dataset/WaterBodies', split='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # 测试通常 BS=1 方便处理

    print("Testing on specific angles:")
    for angle in ['0', '90', '180', '270']:
        evaluate_rotation(model, test_loader, angle_mode=angle, device=device)

    print("Testing on random angles:")
    evaluate_rotation(model, test_loader, angle_mode='random', device=device)