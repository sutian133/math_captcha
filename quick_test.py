#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速测试脚本 - 配套train_v2.py
简单快速地测试模型基本功能
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import logging
from train import FixedCRNN, MathCaptchaDataset

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def quick_test():
    """快速测试模型"""
    print("🚀 快速测试 v2")
    print("=" * 40)
    
    # 1. 检查模型文件
    model_path = Path('checkpoints/best_model_v2.pth')
    if not model_path.exists():
        print("❌ 找不到模型文件: checkpoints/best_model_v2.pth")
        print("💡 请先运行: python train.py")
        return
    
    # 2. 加载模型
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        char_to_idx = checkpoint['char_to_idx']
        idx_to_char = checkpoint['idx_to_char']
        
        model = FixedCRNN(num_classes=len(char_to_idx))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        print(f"✅ 模型加载成功")
        print(f"📊 字符集大小: {len(char_to_idx)}")
        print(f"🔧 设备: {device}")
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 3. 检查测试数据
    test_dir = Path('data/generated/test')
    if not test_dir.exists():
        print("❌ 找不到测试数据目录")
        return
    
    try:
        dataset = MathCaptchaDataset(test_dir, char_to_idx, 'test')
        print(f"✅ 测试数据加载成功: {len(dataset)} 个样本")
    except Exception as e:
        print(f"❌ 测试数据加载失败: {e}")
        return
    
    # 4. 快速预测几个样本
    print(f"\n🧪 测试 5 个随机样本:")
    print("-" * 40)
    
    import random
    test_indices = random.sample(range(len(dataset)), min(5, len(dataset)))
    
    correct = 0
    for i, idx in enumerate(test_indices, 1):
        img, _, true_text, answer = dataset[idx]
        
        # 预测
        img = img.unsqueeze(0).to(device)
        with torch.no_grad():
            log_probs = model(img)
            predictions = model.decode(log_probs, idx_to_char)
        
        predicted = predictions[0] if predictions else ""
        is_correct = predicted == true_text
        if is_correct:
            correct += 1
        
        # 输出结果
        status = "✅" if is_correct else "❌"
        type_icon = "🇨🇳" if any('\u4e00' <= c <= '\u9fff' for c in true_text) else "🔢"
        print(f"{i}. {type_icon} {status} {true_text} → {predicted}")
    
    # 5. 总结
    accuracy = correct / len(test_indices) * 100
    print(f"\n📈 快速测试准确率: {accuracy:.1f}% ({correct}/{len(test_indices)})")
    
    if accuracy >= 80:
        print("🎉 模型表现优秀！")
    elif accuracy >= 50:
        print("✅ 模型表现良好")
    elif accuracy >= 20:
        print("⚠️ 模型表现一般")
    else:
        print("🔴 模型需要重新训练")
    
    print(f"\n💡 运行完整测试: python test_v2.py")

if __name__ == "__main__":
    quick_test() 