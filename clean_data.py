import os
import numpy as np
from PIL import Image
from tqdm import tqdm


def filter_dataset(root_dir):
    masks_dir = os.path.join(root_dir, 'Masks')
    images_dir = os.path.join(root_dir, 'Images')

    all_files = os.listdir(masks_dir)
    valid_files = []

    print("开始清洗数据集 (剔除全黑或全白的 Mask)...")

    for filename in tqdm(all_files):
        mask_path = os.path.join(masks_dir, filename)
        # 读取 Mask
        mask = Image.open(mask_path).convert('L')
        mask_np = np.array(mask)

        # 二值化处理 (0 是背景, 255 是水)
        # 有些 mask 可能是 0/1 或 0/255，统一处理
        is_water = (mask_np > 127).astype(np.int8)

        water_pixels = np.sum(is_water)
        total_pixels = is_water.size
        water_ratio = water_pixels / total_pixels

        # 筛选规则：
        # 1. 剔除全黑 (没有水)
        # 2. 剔除全白 (全是水) - 这一步可选，但通常有助于提高边界学习能力
        # 论文数量 2328 / 2841 ≈ 0.81。
        # 如果只剔除全黑，通常能对齐这个数量。

        if water_pixels > 0 and water_ratio < 1.0:
            # 检查对应的 Image 是否存在
            if os.path.exists(os.path.join(images_dir, filename)):
                valid_files.append(filename)

    print(f"原始文件数: {len(all_files)}")
    print(f"清洗后有效文件数: {len(valid_files)}")
    print(f"预期论文数量: ~2328")

    # 保存有效列表
    with open('valid_images.txt', 'w') as f:
        for item in valid_files:
            f.write("%s\n" % item)

    return len(valid_files)


if __name__ == '__main__':
    # 请修改为你的数据集路径
    DATASET_ROOT = 'E:/PyCharm/Projects/data/Satellite/Water Bodies Dataset'
    filter_dataset(DATASET_ROOT)