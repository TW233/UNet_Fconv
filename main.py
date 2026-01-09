import os
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Subset
from dataset import WaterDataset
from engine import ExperimentEngine
from utils import setup_logger, load_model_class

# === 配置区域 ===
EXPERIMENTS = [
    {
        'name': 'PreCM_Replication_Round',  # 实验名称
        'file_path': 'networks/UNet/unet.py',  # 指向你修改后的 3.35M 小模型文件
        'class_name': 'Unet',
        'batch_size': 16,  # 模型变小了，Batch Size 可以开大，建议 16 或 32
        'epochs': 150
    }
]

DATASET_ROOT = 'E:/PyCharm/Projects/data/Satellite/Water Bodies Dataset'  # 请确认你的数据路径
VALID_LIST_FILE = 'valid_images.txt'  # clean_data.py 生成的列表
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_ROUNDS = 5  # 论文要求的 5 轮实验


def run_round(exp_config, train_dataset, test_dataset, round_idx, train_indices, test_indices, logger):
    """
    执行单轮实验
    train_dataset: 开启了增强的数据集对象
    test_dataset: 关闭了增强的数据集对象
    """
    exp_name = f"{exp_config['name']}_Round{round_idx + 1}"
    save_dir = os.path.join('results', exp_config['name'], f'round_{round_idx + 1}')
    os.makedirs(save_dir, exist_ok=True)

    logger.info(f"--- Round {round_idx + 1} Started ---")

    # 1. 动态加载模型
    ModelClass = load_model_class(exp_config['file_path'], exp_config['class_name'])
    # 注意：这里初始化的是修改后的 32通道 U-Net
    model = ModelClass(in_channels=3, classes=2)

    # 2. 权重初始化 (这是复现论文的关键细节)
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:
            torch.nn.init.normal_(param, mean=0.0, std=0.02)
        elif 'bias' in name:
            torch.nn.init.constant_(param, 0.0)

    # 3. 构建 Subset
    # 关键点：训练集从“增强版”Dataset取，测试集从“纯净版”Dataset取
    # 只要 file_list 顺序一致，indices 就是通用的
    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)

    # 4. DataLoader
    # num_workers 建议设为 4 或 8，加快数据读取
    train_loader = DataLoader(train_subset, batch_size=exp_config['batch_size'],
                              shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    # 测试集 batch_size=1 是标准做法
    test_loader = DataLoader(test_subset, batch_size=1, shuffle=False, num_workers=4)

    # 5. 启动引擎
    engine = ExperimentEngine(model, train_loader, test_loader, DEVICE, save_dir, logger)

    # 训练
    engine.train(epochs=exp_config['epochs'])

    # 评估
    metrics = engine.evaluate()

    # 可选：保存可视化结果
    # engine.visualize(num_samples=5)

    return metrics


def main():
    # 1. 读取有效图片列表
    if not os.path.exists(VALID_LIST_FILE):
        print(f"Error: 找不到 {VALID_LIST_FILE}，请先运行 clean_data.py")
        return

    with open(VALID_LIST_FILE, 'r') as f:
        file_list = [line.strip() for line in f.readlines()]

    total_images = len(file_list)
    print(f"Total Valid Images Loaded: {total_images}")

    # 2. 准备两个 Dataset 对象
    # train_dataset: is_train=True (开启翻转、旋转增强)
    # test_dataset:  is_train=False (仅 Resize 和 Normalize)
    # 假设你已经把图片 resize 改回了 256 (推荐) 或保持 448
    IMG_SIZE = 256

    train_dataset = WaterDataset(DATASET_ROOT, file_list, img_size=IMG_SIZE, is_train=True)
    test_dataset = WaterDataset(DATASET_ROOT, file_list, img_size=IMG_SIZE, is_train=False)

    # 3. 论文规定的训练集数量
    NUM_TRAIN = 1662

    for exp_config in EXPERIMENTS:
        logger = setup_logger(os.path.join('results', exp_config['name']))
        logger.info(f"Experiment Configuration: {exp_config}")

        round_ious = []
        round_rds = []

        for round_idx in range(NUM_ROUNDS):
            # === 每一轮重新随机划分数据集 ===
            indices = list(range(total_images))
            random.shuffle(indices)  # 打乱

            train_indices = indices[:NUM_TRAIN]
            test_indices = indices[NUM_TRAIN:]  # 剩余所有作为测试集

            logger.info(f"Round {round_idx + 1} Split: Train={len(train_indices)}, Test={len(test_indices)}")

            # 运行单轮实验
            metrics = run_round(exp_config, train_dataset, test_dataset, round_idx, train_indices, test_indices, logger)

            if metrics:
                # 记录核心指标：0度下的 IoU 和 RD
                iou_0 = metrics['0']['iou']
                rd_val = metrics['0']['rd']

                round_ious.append(iou_0)
                round_rds.append(rd_val)
                logger.info(f"Round {round_idx + 1} Result -> IoU(0): {iou_0:.2f}, RD: {rd_val:.2f}")

        # === 5轮结束后计算平均值 ===
        avg_iou = np.mean(round_ious)
        std_iou = np.std(round_ious)
        avg_rd = np.mean(round_rds)

        logger.info("=" * 40)
        logger.info(f"FINAL RESULTS ({NUM_ROUNDS} Rounds)")
        logger.info(f"Avg IoU (0°): {avg_iou:.2f} ± {std_iou:.2f}")
        logger.info(f"Avg RD: {avg_rd:.2f}")
        logger.info("=" * 40)


if __name__ == '__main__':
    # 设置随机种子 (可选，为了完全复现可以固定，但论文建议是随机多次)
    # seed = 42
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    main()