#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数学验证码识别系统配置文件
支持中文数字、阿拉伯数字和数学运算符的高精度识别
"""

import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / 'data'
MODELS_ROOT = PROJECT_ROOT / 'models'

# 数据配置
DATA_CONFIG = {
    'train_dir': DATA_ROOT / 'train',
    'test_dir': DATA_ROOT / 'test',
    'generated_dir': DATA_ROOT / 'generated',
    'image_size': (160, 60),  # 宽x高
    'channels': 3,
    'max_length': 8,  # 最大字符序列长度 如: 三加二等于五
}

# 字符集定义 - 支持中文数字、阿拉伯数字、运算符
CHARACTER_SET = {
    # 阿拉伯数字 0-9
    'arabic_numbers': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    
    # 中文数字
    'chinese_numbers': ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九'],
    
    # 阿拉伯运算符
    'arabic_operators': ['+', '-', '×', '÷', '=', '*', '/'],
    
    # 中文运算符
    'chinese_operators': ['加', '减', '乘', '除', '等于', '等', '于'],
    
    # 其他符号
    'symbols': ['?', '？', '(', ')', ' '],
    
    # 特殊token
    'special': ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
}

# 构建完整字符集和映射
ALL_CHARS = []
for category in CHARACTER_SET.values():
    ALL_CHARS.extend(category)

# 字符到索引的映射
CHAR_TO_IDX = {char: idx for idx, char in enumerate(ALL_CHARS)}
IDX_TO_CHAR = {idx: char for char, idx in CHAR_TO_IDX.items()}
NUM_CLASSES = len(ALL_CHARS)

# 模型配置
MODEL_CONFIG = {
    'cnn': {
        'backbone': 'resnet18',
        'feature_dim': 512,
        'dropout': 0.2,
    },
    'lstm': {
        'hidden_size': 256,
        'num_layers': 2,
        'bidirectional': True,
        'dropout': 0.3,
    },
    'transformer': {
        'embed_dim': 256,
        'num_heads': 8,
        'num_layers': 6,
        'ff_dim': 1024,
        'dropout': 0.1,
    },
    'attention': {
        'attention_dim': 256,
        'use_coverage': True,
    }
}

# 训练配置
TRAIN_CONFIG = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'epochs': 100,
    'early_stopping_patience': 10,
    'scheduler': {
        'type': 'CosineAnnealingLR',
        'T_max': 50,
        'eta_min': 1e-6
    },
    'optimizer': 'AdamW',
    'loss_function': 'CrossEntropyLoss',
    'metrics': ['accuracy', 'sequence_accuracy', 'edit_distance']
}

# 数据增强配置
AUGMENTATION_CONFIG = {
    'rotation': {'limit': 5, 'p': 0.3},
    'noise': {'var_limit': (0.01, 0.05), 'p': 0.3},
    'blur': {'blur_limit': 3, 'p': 0.2},
    'brightness': {'limit': 0.2, 'p': 0.3},
    'contrast': {'limit': 0.2, 'p': 0.3},
    'elastic_transform': {'alpha': 1, 'sigma': 50, 'p': 0.2},
    'grid_distortion': {'num_steps': 5, 'distort_limit': 0.3, 'p': 0.2},
}

# 生成器配置
GENERATOR_CONFIG = {
    'num_samples': {
        'train': 50000,
        'val': 5000,
        'test': 5000
    },
    'fonts': [
        'arial.ttf',
        'simhei.ttf', 
        'times.ttf',
        'simsun.ttc',
        'calibri.ttf'
    ],
    'font_sizes': [24, 28, 32, 36, 40],
    'background_colors': [(255, 255, 255), (240, 240, 240), (250, 250, 250)],
    'text_colors': [(0, 0, 0), (50, 50, 50), (30, 30, 30)],
    'noise_intensity': 0.1,
    'interference_lines': 3,
    'equation_types': [
        'arabic_simple',      # 1+2=?
        'chinese_simple',     # 一加二等于？
        'mixed_notation',     # 1+二=？
        'parentheses',        # (1+2)×3=?
        'complex_chinese'     # 三乘以二加一等于？
    ]
}

# 验证规则配置
VALIDATION_CONFIG = {
    'min_accuracy': 0.95,  # 最小准确率要求
    'sequence_accuracy': 0.90,  # 完整序列准确率
    'max_edit_distance': 1,  # 最大编辑距离
}

# 模型保存配置
CHECKPOINT_CONFIG = {
    'save_dir': MODELS_ROOT / 'checkpoints',
    'save_freq': 5,  # 每5个epoch保存一次
    'keep_best': True,
    'metric': 'sequence_accuracy'  # 用于判断最佳模型的指标
}

# 推理配置
INFERENCE_CONFIG = {
    'confidence_threshold': 0.8,
    'use_beam_search': True,
    'beam_width': 3,
    'max_decode_length': 10,
    'use_ensemble': True,
    'tta': True,  # Test Time Augmentation
}

# 日志配置
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'save_dir': PROJECT_ROOT / 'logs',
    'tensorboard_dir': PROJECT_ROOT / 'tensorboard_logs'
}

# 数学表达式验证规则
MATH_RULES = {
    'max_operand': 100,
    'min_operand': 0,
    'allowed_operators': ['+', '-', '×', '÷', '加', '减', '乘', '除'],
    'division_check': True,  # 检查除法是否整除
    'negative_result': False,  # 是否允许负数结果
}

def get_config():
    """获取完整配置"""
    return {
        'data': DATA_CONFIG,
        'model': MODEL_CONFIG,
        'train': TRAIN_CONFIG,
        'augmentation': AUGMENTATION_CONFIG,
        'generator': GENERATOR_CONFIG,
        'validation': VALIDATION_CONFIG,
        'checkpoint': CHECKPOINT_CONFIG,
        'inference': INFERENCE_CONFIG,
        'logging': LOGGING_CONFIG,
        'math_rules': MATH_RULES,
        'characters': {
            'char_to_idx': CHAR_TO_IDX,
            'idx_to_char': IDX_TO_CHAR,
            'num_classes': NUM_CLASSES,
            'character_set': CHARACTER_SET
        }
    }

# 环境变量设置
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 设置GPU
os.environ['PYTHONPATH'] = str(PROJECT_ROOT)

if __name__ == "__main__":
    config = get_config()
    print("数学验证码识别系统配置:")
    print(f"字符类别数: {NUM_CLASSES}")
    print(f"支持的字符: {ALL_CHARS[:20]}...")  # 显示前20个字符
    print(f"图片尺寸: {DATA_CONFIG['image_size']}")
    print(f"批次大小: {TRAIN_CONFIG['batch_size']}")
    print(f"学习率: {TRAIN_CONFIG['learning_rate']}") 