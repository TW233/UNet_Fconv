import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class WaterDataset(Dataset):
    def __init__(self, root_dir, file_list, img_size=448):
        """
        file_list: 包含文件名的列表 (从 valid_images.txt 读取)
        """
        self.root_dir = root_dir
        self.image_list = file_list
        self.images_dir = os.path.join(root_dir, 'Images')
        self.masks_dir = os.path.join(root_dir, 'Masks')
        
        # 强制 Resize 到 448，以适配 PreCM 的硬编码
        self.img_size = img_size 

        self.img_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.mask_transform = transforms.Compose([
            # Mask 必须用 NEAREST 插值，防止出现 0.5 这种小数
            transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.img_transform(image)
        mask = self.mask_transform(mask)
        
        # 二值化
        mask = (mask > 0.5).float()

        return image, mask