import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import WaterDataset
# 引入你的模型，这里假设你用的是 Fconv 版本的 UNet
from networks import unet_fconv  # 请确认文件名和类名匹配
import os


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 数据准备
    train_dataset = WaterDataset(root_dir='../data/Satellite/Water Bodies Dataset', split='train')
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)

    # 2. 模型初始化
    # 注意：PreCM 论文为了公平对比，把通道数减半了(32起始)，你的代码里如果默认是64要注意修改
    # 这里我们直接实例化你的 Fconv UNet
    model = unet_fconv(in_channels=3, classes=2).to(device)

    # 权重初始化 (Gaussian, 论文提到)
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:
            nn.init.normal_(param, mean=0.0, std=0.02)

    # 3. 优化器和损失
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss()

    print("Start Training...")
    for epoch in range(150):
        model.train()
        epoch_loss = 0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device).long().squeeze(1)  # CrossEntropy 需要 target 为 long 且 shape (N, H, W)

            optimizer.zero_grad()
            outputs = model(images)  # 你的模型输出应该是 (N, 2, H, W)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch + 1}/150], Loss: {epoch_loss / len(train_loader):.4f}")

        # 建议每隔一定 epoch 保存模型
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), f'checkpoints/fconv_unet_epoch_{epoch + 1}.pth')


if __name__ == '__main__':
    # 确保文件夹存在
    os.makedirs('checkpoints', exist_ok=True)
    train()