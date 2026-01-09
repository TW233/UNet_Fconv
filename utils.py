import importlib.util
import os
import torch
import random
import logging


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
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(save_dir, 'log.txt')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()


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