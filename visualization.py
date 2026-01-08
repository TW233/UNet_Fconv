import matplotlib.pyplot as plt
import torch
import numpy as np
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from dataset import WaterDataset
from networks import unet_fconv
import torch.nn.functional as F
import random

# ... 引入前面的模型和数据处理 ...

def visualize_difference(model, image, mask, device):
    # 取一张图
    img_tensor = image.unsqueeze(0).to(device)  # (1, 3, H, W)

    # 1. 原始预测
    out_origin = model(img_tensor)
    pred_origin = torch.argmax(out_origin, dim=1).float()

    # 2. 旋转预测 (例如旋转 90 度)
    angle = 90
    img_rot = TF.rotate(img_tensor, angle)
    out_rot = model(img_rot)
    pred_rot = torch.argmax(out_rot, dim=1).float()

    # 3. 转回
    pred_back = TF.rotate(pred_rot, -angle)

    # 4. 计算差异 (0: 无差异, 1: 有差异)
    diff = torch.abs(pred_origin - pred_back).cpu().numpy().squeeze()

    # 绘图
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 4, 1);
    plt.title("Input");
    plt.imshow(image.permute(1, 2, 0))
    plt.subplot(1, 4, 2);
    plt.title("GT");
    plt.imshow(mask.squeeze(), cmap='gray')
    plt.subplot(1, 4, 3);
    plt.title("Pred (0 deg)");
    plt.imshow(pred_origin.cpu().squeeze(), cmap='gray')
    plt.subplot(1, 4, 4);
    plt.title("Difference Map");
    plt.imshow(diff, cmap='hot')  # hot colormap 高亮差异
    plt.show()
    # 可以保存图片
    plt.savefig('vis_result.png')

# 调用方法同 evaluate.py，取一个 batch 进行可视化