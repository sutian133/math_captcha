#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成更多数学验证码训练数据
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import os
import json
from pathlib import Path

def test_chinese_font():
    """测试系统是否支持中文字体"""
    chinese_font_paths = [
        'C:/Windows/Fonts/simhei.ttf',
        'C:/Windows/Fonts/simsun.ttc', 
        'C:/Windows/Fonts/msyh.ttc',
        'C:/Windows/Fonts/kaiti.ttf',
    ]
    
    for font_path in chinese_font_paths:
        if os.path.exists(font_path):
            return font_path
    return None

def create_math_image(text, size=(160, 60), chinese_font_path=None):
    """创建数学验证码图片"""
    width, height = size
    
    # 创建白色背景
    img = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # 检查文本是否包含中文
    has_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
    
    # 选择合适的字体
    font = None
    try:
        if has_chinese and chinese_font_path and os.path.exists(chinese_font_path):
            # 使用中文字体，随机字体大小
            font_size = random.choice([20, 22, 24, 26, 28])
            font = ImageFont.truetype(chinese_font_path, font_size)
        else:
            # 使用英文字体
            english_fonts = [
                'C:/Windows/Fonts/arial.ttf',
                'C:/Windows/Fonts/calibri.ttf',
                'C:/Windows/Fonts/times.ttf'
            ]
            
            for font_path in english_fonts:
                if os.path.exists(font_path):
                    font_size = random.choice([20, 22, 24, 26, 28])
                    font = ImageFont.truetype(font_path, font_size)
                    break
        
        if font is None:
            font = ImageFont.load_default()
            
    except Exception as e:
        print(f"字体加载失败: {e}")
        font = ImageFont.load_default()
    
    # 如果有中文但没有中文字体，跳过这个文本
    if has_chinese and chinese_font_path is None:
        return None
    
    # 计算文本位置（居中+随机偏移）
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except:
        text_width = len(text) * 15
        text_height = 25
    
    # 基础居中位置
    base_x = (width - text_width) // 2
    base_y = (height - text_height) // 2
    
    # 添加随机偏移（增加数据多样性）
    x_offset = random.randint(-5, 5)
    y_offset = random.randint(-3, 3)
    
    x = max(5, min(base_x + x_offset, width - text_width - 5))
    y = max(5, min(base_y + y_offset, height - text_height - 5))
    
    # 随机文本颜色（深色系）
    text_color = random.choice([
        (0, 0, 0),           # 黑色
        (30, 30, 30),        # 深灰
        (50, 50, 50),        # 灰色
        (20, 20, 80),        # 深蓝
        (80, 20, 20),        # 深红
    ])
    
    # 绘制文本
    draw.text((x, y), text, font=font, fill=text_color)
    
    # 添加轻微的背景噪声（提高鲁棒性）
    for _ in range(random.randint(1, 4)):
        x1 = random.randint(0, width//3)
        y1 = random.randint(0, height)
        x2 = random.randint(2*width//3, width)
        y2 = random.randint(0, height)
        color = (random.randint(220, 240), random.randint(220, 240), random.randint(220, 240))
        draw.line([(x1, y1), (x2, y2)], fill=color, width=1)
    
    return np.array(img)

def generate_diverse_equations(num_samples=1000, chinese_ratio=0.3):
    """生成多样化的数学算式"""
    equations = []
    
    chinese_count = int(num_samples * chinese_ratio)
    english_count = num_samples - chinese_count
    
    print(f"生成 {english_count} 个英文算式和 {chinese_count} 个中文算式...")
    
    # 生成英文数字算式（扩展范围）
    for _ in range(english_count):
        # 随机选择操作类型
        op_type = random.choice(['+', '-', '*', '/'])
        
        if op_type == '+':
            a = random.randint(0, 9)
            b = random.randint(0, 9)
            answer = a + b
            eq = f"{a}+{b}=?"
        elif op_type == '-':
            a = random.randint(1, 9)
            b = random.randint(0, a)  # 确保结果非负
            answer = a - b
            eq = f"{a}-{b}=?"
        elif op_type == '*':
            a = random.randint(1, 9)
            b = random.randint(1, 9)
            answer = a * b
            eq = f"{a}*{b}=?"
        else:  # 除法
            # 确保整除
            answer = random.randint(1, 9)
            b = random.randint(1, 9)
            a = answer * b
            eq = f"{a}/{b}=?"
        
        equations.append((eq, answer))
    
    # 生成中文数字算式
    chinese_numbers = ['一', '二', '三', '四', '五', '六', '七', '八', '九']
    
    for _ in range(chinese_count):
        # 随机选择操作类型
        op_type = random.choice(['加', '减'])
        
        if op_type == '加':
            a = random.randint(1, 7)  # 限制范围避免结果过大
            b = random.randint(1, 7)
            answer = a + b
            eq = f"{chinese_numbers[a-1]}加{chinese_numbers[b-1]}等于?"
        else:  # 减法
            a = random.randint(2, 8)
            b = random.randint(1, a-1)  # 确保结果为正
            answer = a - b
            eq = f"{chinese_numbers[a-1]}减{chinese_numbers[b-1]}等于?"
        
        equations.append((eq, answer))
    
    # 打乱顺序
    random.shuffle(equations)
    return equations

def generate_large_dataset(output_dir, num_samples=1000):
    """生成大规模数据集"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 检测中文字体支持
    chinese_font_path = test_chinese_font()
    support_chinese = chinese_font_path is not None
    
    print(f"正在生成 {num_samples} 个验证码图片样本...")
    if support_chinese:
        print(f"✅ 将生成中英文混合验证码")
        print(f"🎨 使用中文字体: {chinese_font_path}")
    else:
        print(f"⚠️ 只生成英文数字验证码")
    
    equations = generate_diverse_equations(num_samples, chinese_ratio=0.3 if support_chinese else 0.0)
    labels = []
    skipped = 0
    
    for i, (equation, answer) in enumerate(equations):
        # 生成图片
        img = create_math_image(equation, chinese_font_path=chinese_font_path)
        
        # 如果图片生成失败，跳过
        if img is None:
            skipped += 1
            continue
        
        # 保存图片
        filename = f"captcha_{i:05d}.png"
        img_path = output_path / filename
        
        cv2.imwrite(str(img_path), img)
        
        # 记录标签
        labels.append({
            'filename': filename,
            'equation': equation,
            'answer': answer,
            'characters': list(equation),
            'type': 'chinese' if any('\u4e00' <= char <= '\u9fff' for char in equation) else 'english'
        })
        
        if (i + 1) % 100 == 0:
            print(f"已生成 {len(labels)}/{num_samples} 个有效样本")
    
    # 保存标签文件
    with open(output_path / 'labels.json', 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 数据集生成完成！保存在 {output_path}")
    print(f"   有效图片: {len(labels)} 个")
    print(f"   跳过样本: {skipped} 个")
    
    return len(labels)

if __name__ == "__main__":
    print("🚀 大规模数学验证码数据生成器")
    print("=" * 60)
    
    # 清理旧数据
    import shutil
    old_train = Path('data/generated/train')
    old_test = Path('data/generated/test')
    
    if old_train.exists():
        print(f"🗑️ 清理旧训练数据...")
        shutil.rmtree(old_train)
    
    if old_test.exists():
        print(f"🗑️ 清理旧测试数据...")
        shutil.rmtree(old_test)
    
    # 生成新的大规模数据集
    print(f"\n📁 生成新的训练数据集...")
    train_samples = generate_large_dataset('data/generated/train', 8000)  # 8000个训练样本
    
    print(f"\n📁 生成新的测试数据集...")
    test_samples = generate_large_dataset('data/generated/test', 2000)   # 2000个测试样本
    
    # 汇总报告
    print(f"\n📊 新数据集汇总:")
    print(f"   训练样本: {train_samples}")
    print(f"   测试样本: {test_samples}")
    print(f"   总计样本: {train_samples + test_samples}")
    print(f"\n✅ 大规模数据集生成完成！")
    print(f"💡 建议使用更高学习率训练: python train.py --epochs 50 --lr 0.01 --batch_size 8 --from_scratch")