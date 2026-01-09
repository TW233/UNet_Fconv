import os
import torch
import random
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class WaterDataset(Dataset):
    def __init__(self, root_dir, file_list, img_size=256, is_train=True):
        # === 修改点 1: 默认 img_size 改为 256 ===
        self.root_dir = root_dir
        self.image_list = file_list
        self.images_dir = os.path.join(root_dir, 'Images')
        self.masks_dir = os.path.join(root_dir, 'Masks')
        self.img_size = img_size
        self.is_train = is_train
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Resize
        image = TF.resize(image, (self.img_size, self.img_size))
        mask = TF.resize(mask, (self.img_size, self.img_size), interpolation=transforms.InterpolationMode.NEAREST)

        # === 修改点 2: 仅保留翻转增强 (复现 Table I Baseline) ===
        if self.is_train:
            # 随机水平翻转
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # 随机垂直翻转
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            # --- 暂时注释掉旋转增强，为了先对齐 Table I 的 80.23% ---
            # 只有当你想要复现 Table II (84%) 时，才解开下面这段代码
            # if random.random() > 0.5:
            #     angle = random.choice([90, 180, 270])
            #     image = TF.rotate(image, angle)
            #     mask = TF.rotate(mask, angle)

        image = TF.to_tensor(image)
        image = self.norm(image)
        mask = TF.to_tensor(mask)
        mask = (mask > 0.5).float()

        return image, mask