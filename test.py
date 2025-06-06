#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配套train_v2.py的测试脚本
测试修复后的数学验证码识别模型
"""

import torch
import cv2
import numpy as np
import json
from pathlib import Path
import logging
import random
from train import FixedCRNN, MathCaptchaDataset, analyze_dataset_chars, create_char_mapping

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_v2(checkpoint_path='checkpoints/best_model_v2.pth'):
    """加载训练好的v2模型"""
    if not Path(checkpoint_path).exists():
        logger.error(f"❌ 模型文件不存在: {checkpoint_path}")
        logger.info("💡 请先运行训练脚本: python train.py")
        return None, None, None, False

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        char_to_idx = checkpoint['char_to_idx']
        idx_to_char = checkpoint['idx_to_char']
        
        model = FixedCRNN(num_classes=len(char_to_idx))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        logger.info(f"✅ 模型加载成功: {checkpoint_path}")
        logger.info(f"📊 字符集大小: {len(char_to_idx)}")
        logger.info(f"🎯 训练轮次: {checkpoint.get('epoch', 'N/A')}")
        logger.info(f"📈 最佳损失: {checkpoint.get('loss', 'N/A'):.4f}")
        
        return model, char_to_idx, idx_to_char, True
        
    except Exception as e:
        logger.error(f"❌ 模型加载失败: {e}")
        return None, None, None, False

def preprocess_single_image(image_path, target_size=(160, 60)):
    """预处理单张图片"""
    img = cv2.imread(str(image_path))
    if img is None:
        logger.error(f"无法读取图片: {image_path}")
        return None
    
    # 调整大小
    img = cv2.resize(img, target_size)
    
    # 归一化
    img = img.astype(np.float32) / 255.0
    
    # 转换为tensor (H, W, C) -> (C, H, W)
    img_tensor = torch.from_numpy(img).permute(2, 0, 1)
    
    # 添加batch维度
    img_tensor = img_tensor.unsqueeze(0)  # (1, C, H, W)
    
    return img_tensor

def predict_single_image(model, image_path, idx_to_char, device):
    """预测单张图片"""
    img_tensor = preprocess_single_image(image_path)
    if img_tensor is None:
        return ""
    
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        log_probs = model(img_tensor)  # (seq_len, batch_size, num_classes)
        predictions = model.decode(log_probs, idx_to_char)
    
    return predictions[0] if predictions else ""

def test_dataset_v2(model, char_to_idx, idx_to_char, device, data_dir='data/generated/test', num_samples=20):
    """测试数据集"""
    if not Path(data_dir).exists():
        logger.error(f"❌ 测试数据目录不存在: {data_dir}")
        return 0.0
    
    dataset = MathCaptchaDataset(data_dir, char_to_idx, 'test')
    
    # 随机选择样本测试
    total_samples = len(dataset)
    test_indices = random.sample(range(total_samples), min(num_samples, total_samples))
    
    correct = 0
    total = len(test_indices)
    
    logger.info(f"🧪 测试 {total} 个样本...")
    logger.info("-" * 60)
    
    for i, idx in enumerate(test_indices, 1):
        img, char_indices, true_text, answer = dataset[idx]
        
        # 预处理图片
        img = img.unsqueeze(0).to(device)  # 添加batch维度
        
        # 预测
        with torch.no_grad():
            log_probs = model(img)
            predictions = model.decode(log_probs, idx_to_char)
        
        predicted_text = predictions[0] if predictions else ""
        is_correct = predicted_text == true_text
        
        if is_correct:
            correct += 1
        
        # 显示结果
        type_icon = "🇨🇳" if any('\u4e00' <= char <= '\u9fff' for char in true_text) else "🔢"
        status = "✅" if is_correct else "❌"
        
        logger.info(f"{i:2d}. {type_icon} {status}")
        logger.info(f"    真实: {true_text}")
        logger.info(f"    预测: {predicted_text}")
        logger.info(f"    答案: {answer}")
        
        # 如果预测错误，显示详细信息
        if not is_correct:
            logger.info(f"    🔍 字符差异分析:")
            logger.info(f"       真实长度: {len(true_text)}, 预测长度: {len(predicted_text)}")
            for j, (t, p) in enumerate(zip(true_text, predicted_text)):
                if t != p:
                    logger.info(f"       位置{j}: '{t}' vs '{p}'")
        logger.info("")
    
    accuracy = correct / total * 100
    
    logger.info("📊 测试结果汇总:")
    logger.info("-" * 60)
    logger.info(f"✅ 正确预测: {correct}")
    logger.info(f"❌ 错误预测: {total - correct}")
    logger.info(f"📈 准确率: {accuracy:.2f}% ({correct}/{total})")
    
    return accuracy

def analyze_model_performance(model, char_to_idx, idx_to_char, device, data_dir='data/generated/test'):
    """分析模型性能"""
    logger.info("\n🔍 模型性能详细分析:")
    logger.info("=" * 60)
    
    if not Path(data_dir).exists():
        logger.error(f"❌ 测试数据目录不存在: {data_dir}")
        return
    
    dataset = MathCaptchaDataset(data_dir, char_to_idx, 'test')
    
    # 按类型分类统计
    chinese_correct = 0
    chinese_total = 0
    english_correct = 0
    english_total = 0
    
    # 测试所有样本
    for i in range(min(100, len(dataset))):  # 最多测试100个样本
        img, char_indices, true_text, answer = dataset[i]
        
        # 预测
        img = img.unsqueeze(0).to(device)
        with torch.no_grad():
            log_probs = model(img)
            predictions = model.decode(log_probs, idx_to_char)
        
        predicted_text = predictions[0] if predictions else ""
        is_correct = predicted_text == true_text
        
        # 分类统计
        if any('\u4e00' <= char <= '\u9fff' for char in true_text):
            chinese_total += 1
            if is_correct:
                chinese_correct += 1
        else:
            english_total += 1
            if is_correct:
                english_correct += 1
    
    # 输出分析结果
    chinese_acc = chinese_correct / chinese_total * 100 if chinese_total > 0 else 0
    english_acc = english_correct / english_total * 100 if english_total > 0 else 0
    total_acc = (chinese_correct + english_correct) / (chinese_total + english_total) * 100
    
    logger.info(f"🇨🇳 中文验证码:")
    logger.info(f"   样本数: {chinese_total}")
    logger.info(f"   正确数: {chinese_correct}")
    logger.info(f"   准确率: {chinese_acc:.2f}%")
    
    logger.info(f"\n🔢 英文验证码:")
    logger.info(f"   样本数: {english_total}")
    logger.info(f"   正确数: {english_correct}")
    logger.info(f"   准确率: {english_acc:.2f}%")
    
    logger.info(f"\n📊 整体性能:")
    logger.info(f"   总样本: {chinese_total + english_total}")
    logger.info(f"   总正确: {chinese_correct + english_correct}")
    logger.info(f"   总准确率: {total_acc:.2f}%")

def show_prediction_samples(model, char_to_idx, idx_to_char, device, data_dir='data/generated/test'):
    """显示预测样本（用于调试）"""
    logger.info("\n🖼️ 预测样本展示:")
    logger.info("-" * 40)
    
    if not Path(data_dir).exists():
        logger.error(f"❌ 测试数据目录不存在: {data_dir}")
        return
    
    dataset = MathCaptchaDataset(data_dir, char_to_idx, 'test')
    
    # 选择不同类型的样本
    chinese_samples = []
    english_samples = []
    
    for i in range(min(50, len(dataset))):
        _, _, true_text, _ = dataset[i]
        if any('\u4e00' <= char <= '\u9fff' for char in true_text):
            if len(chinese_samples) < 3:
                chinese_samples.append(i)
        else:
            if len(english_samples) < 3:
                english_samples.append(i)
        
        if len(chinese_samples) >= 3 and len(english_samples) >= 3:
            break
    
    # 显示中文样本
    if chinese_samples:
        logger.info("\n🇨🇳 中文样本:")
        for i, idx in enumerate(chinese_samples, 1):
            img, _, true_text, answer = dataset[idx]
            img = img.unsqueeze(0).to(device)
            
            with torch.no_grad():
                log_probs = model(img)
                predictions = model.decode(log_probs, idx_to_char)
            
            predicted = predictions[0] if predictions else ""
            status = "✅" if predicted == true_text else "❌"
            logger.info(f"  {i}. {true_text} → {predicted} {status}")
    
    # 显示英文样本
    if english_samples:
        logger.info("\n🔢 英文样本:")
        for i, idx in enumerate(english_samples, 1):
            img, _, true_text, answer = dataset[idx]
            img = img.unsqueeze(0).to(device)
            
            with torch.no_grad():
                log_probs = model(img)
                predictions = model.decode(log_probs, idx_to_char)
            
            predicted = predictions[0] if predictions else ""
            status = "✅" if predicted == true_text else "❌"
            logger.info(f"  {i}. {true_text} → {predicted} {status}")

def test_single_image_file(image_path):
    """测试单个图片文件"""
    logger.info(f"\n🖼️ 测试单张图片: {image_path}")
    logger.info("-" * 40)
    
    # 加载模型
    model, char_to_idx, idx_to_char, success = load_model_v2()
    if not success:
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 预测
    prediction = predict_single_image(model, image_path, idx_to_char, device)
    
    logger.info(f"📝 预测结果: {prediction}")
    
    return prediction

def main():
    """主测试函数"""
    logger.info("🚀 数学验证码识别模型测试 v2")
    logger.info("=" * 60)
    
    # 加载模型
    model, char_to_idx, idx_to_char, success = load_model_v2()
    if not success:
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    logger.info(f"🔧 使用设备: {device}")
    
    # 显示预测样本
    show_prediction_samples(model, char_to_idx, idx_to_char, device)
    
    # 测试数据集
    accuracy = test_dataset_v2(model, char_to_idx, idx_to_char, device, num_samples=20)
    
    # 性能分析
    analyze_model_performance(model, char_to_idx, idx_to_char, device)
    
    # 评估结果
    logger.info(f"\n💡 评估结果:")
    if accuracy >= 80:
        logger.info("🎉 模型表现优秀！识别准确率很高")
    elif accuracy >= 60:
        logger.info("✅ 模型表现良好，可以继续优化")
    elif accuracy >= 30:
        logger.info("⚠️ 模型表现一般，建议调整训练参数")
    else:
        logger.info("🔴 模型表现较差，建议重新训练")
        logger.info("💡 可能的问题：")
        logger.info("   - 序列长度不匹配")
        logger.info("   - 学习率设置不当")
        logger.info("   - 训练数据不足")
        logger.info("   - 模型架构需要调整")

if __name__ == "__main__":
    main() 