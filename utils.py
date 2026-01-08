import importlib.util
import os
import torch
import logging
import sys

def load_model_class(file_path, class_name='Unet'):
    """
    动态加载带特殊字符（如括号）的文件中的类
    """
    spec = importlib.util.spec_from_file_location("dynamic_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)

def setup_logger(save_dir):
    """设置日志，同时输出到控制台和文件"""
    
    # ================= 核心修复 =================
    # 1. 必须先创建目录，否则 FileHandler 会报 FileNotFoundError
    os.makedirs(save_dir, exist_ok=True)
    # ===========================================

    # 获取根 Logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 2. 清除之前的 handlers
    # 如果不清除，跑完实验A再跑实验B，日志会同时写到 A 和 B 的 log.txt 里
    if logger.hasHandlers():
        logger.handlers.clear()

    # 设置格式
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    # 文件 Handler (写入 log.txt)
    file_handler = logging.FileHandler(os.path.join(save_dir, 'log.txt'), mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 控制台 Handler (输出到屏幕)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger

def calculate_metrics(pred, target):
    """计算 IOU, mIOU, DICE"""
    smooth = 1e-5
    # pred, target: (N, H, W) 0或1
    tp = ((pred == 1) & (target == 1)).sum().item()
    fp = ((pred == 1) & (target == 0)).sum().item()
    fn = ((pred == 0) & (target == 1)).sum().item()
    tn = ((pred == 0) & (target == 0)).sum().item()
    
    iou = tp / (tp + fp + fn + smooth)
    dice = 2 * tp / (2 * tp + fp + fn + smooth)
    iou_bg = tn / (tn + fp + fn + smooth)
    miou = (iou + iou_bg) / 2
    
    return iou * 100, miou * 100, dice * 100