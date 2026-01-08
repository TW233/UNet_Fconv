import os
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Subset
from dataset import WaterDataset
from engine import ExperimentEngine
from utils import setup_logger, load_model_class

# ================= 配置 =================
EXPERIMENTS = [
    {
        'name': 'Baseline_Standard_Unet',
        'file_path': 'networks/UNet/unet.py', 
        'class_name': 'Unet',
        'batch_size': 4, # 显存允许的话，建议改为 8 或 16
        'epochs': 150
    }
    # 你可以在这里添加 PreCM 的配置进行对比
]

DATASET_ROOT = './data/Water Bodies Dataset'
VALID_LIST_FILE = 'valid_images.txt' # 对应第一步生成的文件
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_ROUNDS = 5 # 论文里的“重复5次”
# =======================================

def run_experiment_round(exp_config, full_dataset, round_idx, train_indices, test_indices, logger):
    exp_name = f"{exp_config['name']}_Round{round_idx+1}"
    save_dir = os.path.join('results', exp_config['name'], f'round_{round_idx+1}')
    os.makedirs(save_dir, exist_ok=True)
    
    # 重新初始化 Logger
    # 注意：为了简单，这里你可以复用 logger 或者创建新的 file handler
    logger.info(f"--- Starting Round {round_idx+1}/{NUM_ROUNDS} ---")
    logger.info(f"Train set: {len(train_indices)}, Test set: {len(test_indices)}")
    
    # 1. 动态加载模型 (每一轮都要重新初始化，确保权重重置)
    try:
        ModelClass = load_model_class(exp_config['file_path'], exp_config['class_name'])
        model = ModelClass(in_channels=3, classes=2)
        
        # 显式初始化 (论文提到 Gaussian initialization)
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() > 1:
                torch.nn.init.normal_(param, mean=0.0, std=0.02)
                
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None

    # 2. 创建 DataLoader
    train_subset = Subset(full_dataset, train_indices)
    test_subset = Subset(full_dataset, test_indices)
    
    train_loader = DataLoader(train_subset, batch_size=exp_config['batch_size'], 
                              shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_subset, batch_size=1, shuffle=False, num_workers=4)
    
    # 3. 运行引擎
    engine = ExperimentEngine(model, train_loader, test_loader, DEVICE, save_dir, logger)
    
    # 训练
    engine.train(epochs=exp_config['epochs'])
    
    # 评估
    metrics = engine.evaluate()
    
    # 保存这一轮的结果
    return metrics

def main():
    # 0. 读取清洗后的文件列表
    if not os.path.exists(VALID_LIST_FILE):
        print("请先运行 clean_data.py 生成 valid_images.txt")
        return

    with open(VALID_LIST_FILE, 'r') as f:
        file_list = [line.strip() for line in f.readlines()]
    
    print(f"加载了 {len(file_list)} 张有效图片")
    
    # 初始化数据集对象
    full_dataset = WaterDataset(DATASET_ROOT, file_list, img_size=448)
    
    for exp_config in EXPERIMENTS:
        logger = setup_logger(os.path.join('results', exp_config['name']))
        logger.info(f"Start Experiment Group: {exp_config['name']}")
        
        round_metrics = {'iou': [], 'rd': []}
        
        for round_idx in range(NUM_ROUNDS):
            # === 核心：每一轮重新随机划分 ===
            # 论文: 70% 训练 (这里我们假设剩余30%测试，或者按论文说的随机抽70%做啥)
            # 标准做法：Shuffle -> Split
            total_size = len(file_list)
            indices = list(range(total_size))
            random.shuffle(indices) # 随机打乱
            
            split = int(np.floor(0.7 * total_size))
            train_indices = indices[:split]
            test_indices = indices[split:]
            
            # 运行单轮实验
            metrics = run_experiment_round(exp_config, full_dataset, round_idx, train_indices, test_indices, logger)
            
            if metrics:
                # 记录 Baseline (0度) 的 IOU 和 RD (通常关注0度和Random)
                # 这里记录 0 度 IOU
                round_metrics['iou'].append(metrics['0']['iou'])
                round_metrics['rd'].append(metrics['0']['rd'])
        
        # 计算 5 轮平均值
        avg_iou = np.mean(round_metrics['iou'])
        avg_rd = np.mean(round_metrics['rd'])
        
        logger.info(f"============================================")
        logger.info(f"Experiment {exp_config['name']} Final Result (Avg over 5 rounds):")
        logger.info(f"Avg IOU (0 deg): {avg_iou:.2f}")
        logger.info(f"Avg RD (0 deg): {avg_rd:.2f}")
        logger.info(f"============================================")

if __name__ == '__main__':
    # 确保随机性可复现，也可以不设
    random.seed(42)
    torch.manual_seed(42)
    main()