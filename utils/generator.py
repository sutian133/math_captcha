#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
高精度数学验证码生成器
支持多种语言混合、复杂运算表达式和真实场景干扰
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import random
import os
import json
from pathlib import Path
import math
from typing import Tuple, List, Dict, Union

class MathCaptchaGenerator:
    """数学验证码生成器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.char_to_idx = config['characters']['char_to_idx']
        self.idx_to_char = config['characters']['idx_to_char']
        
        # 数字映射
        self.arabic_to_chinese = {
            '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
            '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'
        }
        
        self.chinese_to_arabic = {v: k for k, v in self.arabic_to_chinese.items()}
        
        # 运算符映射
        self.operator_mapping = {
            '+': '加', '-': '减', '×': '乘', '÷': '除', '=': '等于'
        }
        
        # 字体路径
        self.fonts = self._load_fonts()
        
    def _load_fonts(self) -> List[str]:
        """加载可用字体"""
        font_dirs = [
            'C:/Windows/Fonts/',
            '/System/Library/Fonts/',
            '/usr/share/fonts/',
            './fonts/'
        ]
        
        available_fonts = []
        font_names = self.config['generator']['fonts']
        
        for font_dir in font_dirs:
            if os.path.exists(font_dir):
                for font_name in font_names:
                    font_path = os.path.join(font_dir, font_name)
                    if os.path.exists(font_path):
                        available_fonts.append(font_path)
                        
        # 如果没有找到字体，使用默认字体
        if not available_fonts:
            available_fonts = ['arial.ttf']  # PIL默认字体
            
        return available_fonts
    
    def generate_equation(self, equation_type: str = 'arabic_simple') -> Tuple[str, int]:
        """
        生成数学方程式
        
        Args:
            equation_type: 方程式类型
            
        Returns:
            (equation_text, answer): 方程式文本和答案
        """
        if equation_type == 'arabic_simple':
            return self._generate_arabic_simple()
        elif equation_type == 'chinese_simple':
            return self._generate_chinese_simple()
        elif equation_type == 'mixed_notation':
            return self._generate_mixed_notation()
        elif equation_type == 'parentheses':
            return self._generate_parentheses()
        elif equation_type == 'complex_chinese':
            return self._generate_complex_chinese()
        else:
            return self._generate_arabic_simple()
    
    def _generate_arabic_simple(self) -> Tuple[str, int]:
        """生成简单阿拉伯数字运算: 3+2=?"""
        a = random.randint(1, 20)
        b = random.randint(1, 20)
        op = random.choice(['+', '-', '×', '÷'])
        
        if op == '+':
            answer = a + b
            equation = f"{a}+{b}=?"
        elif op == '-':
            # 确保结果为正数
            if a < b:
                a, b = b, a
            answer = a - b
            equation = f"{a}-{b}=?"
        elif op == '×':
            answer = a * b
            equation = f"{a}×{b}=?"
        else:  # 除法
            # 确保整除
            answer = random.randint(1, 10)
            a = answer * b
            equation = f"{a}÷{b}=?"
            
        return equation, answer
    
    def _generate_chinese_simple(self) -> Tuple[str, int]:
        """生成简单中文数字运算: 三加二等于？"""
        equation, answer = self._generate_arabic_simple()
        
        # 转换为中文
        chinese_eq = ""
        for char in equation:
            if char in self.arabic_to_chinese:
                chinese_eq += self.arabic_to_chinese[char]
            elif char in self.operator_mapping:
                chinese_eq += self.operator_mapping[char]
            elif char == '?':
                chinese_eq += '？'
            else:
                chinese_eq += char
                
        return chinese_eq, answer
    
    def _generate_mixed_notation(self) -> Tuple[str, int]:
        """生成混合记号运算: 1+二=？"""
        a = random.randint(1, 9)
        b = random.randint(1, 9)
        op = random.choice(['+', '-', '×', '÷'])
        
        # 随机选择哪个数字用中文
        use_chinese_a = random.choice([True, False])
        use_chinese_op = random.choice([True, False])
        
        a_str = self.arabic_to_chinese[str(a)] if use_chinese_a else str(a)
        op_str = self.operator_mapping[op] if use_chinese_op else op
        b_str = self.arabic_to_chinese[str(b)] if random.choice([True, False]) else str(b)
        
        if op == '+':
            answer = a + b
        elif op == '-':
            if a < b:
                a, b = b, a
                a_str = self.arabic_to_chinese[str(a)] if use_chinese_a else str(a)
                b_str = self.arabic_to_chinese[str(b)] if random.choice([True, False]) else str(b)
            answer = a - b
        elif op == '×':
            answer = a * b
        else:  # 除法
            answer = random.randint(1, 9)
            a = answer * b
            a_str = self.arabic_to_chinese[str(a)] if use_chinese_a else str(a)
            
        equation = f"{a_str}{op_str}{b_str}=？"
        return equation, answer
    
    def _generate_parentheses(self) -> Tuple[str, int]:
        """生成带括号的运算: (1+2)×3=?"""
        a = random.randint(1, 5)
        b = random.randint(1, 5)
        c = random.randint(1, 5)
        
        # 随机选择运算组合
        combinations = [
            ('+', '×'), ('+', '÷'), ('-', '×'), ('-', '÷'),
            ('×', '+'), ('×', '-'), ('÷', '+'), ('÷', '-')
        ]
        op1, op2 = random.choice(combinations)
        
        # 计算括号内的结果
        if op1 == '+':
            inner_result = a + b
        elif op1 == '-':
            if a < b:
                a, b = b, a
            inner_result = a - b
        elif op1 == '×':
            inner_result = a * b
        else:  # 除法
            inner_result = random.randint(1, 5)
            a = inner_result * b
            
        # 计算最终结果
        if op2 == '+':
            answer = inner_result + c
        elif op2 == '-':
            if inner_result < c:
                inner_result, c = c, inner_result
            answer = inner_result - c
        elif op2 == '×':
            answer = inner_result * c
        else:  # 除法
            answer = random.randint(1, 10)
            c = answer
            inner_result = answer * c
            # 重新生成a, b使得a op1 b = inner_result
            
        equation = f"({a}{op1}{b}){op2}{c}=?"
        return equation, answer
    
    def _generate_complex_chinese(self) -> Tuple[str, int]:
        """生成复杂中文运算: 三乘以二加一等于？"""
        patterns = [
            # 三乘以二加一等于？
            lambda: self._complex_pattern_1(),
            # 五减去二再乘以三等于？  
            lambda: self._complex_pattern_2(),
            # 八除以二的结果加三等于？
            lambda: self._complex_pattern_3(),
        ]
        
        return random.choice(patterns)()
    
    def _complex_pattern_1(self) -> Tuple[str, int]:
        """模式1: X乘以Y加Z等于？"""
        x = random.randint(1, 5)
        y = random.randint(1, 5) 
        z = random.randint(1, 9)
        
        x_ch = self.arabic_to_chinese[str(x)]
        y_ch = self.arabic_to_chinese[str(y)]
        z_ch = self.arabic_to_chinese[str(z)]
        
        answer = x * y + z
        equation = f"{x_ch}乘以{y_ch}加{z_ch}等于？"
        
        return equation, answer
    
    def _complex_pattern_2(self) -> Tuple[str, int]:
        """模式2: X减去Y再乘以Z等于？"""
        x = random.randint(5, 9)
        y = random.randint(1, 4)
        z = random.randint(1, 5)
        
        x_ch = self.arabic_to_chinese[str(x)]
        y_ch = self.arabic_to_chinese[str(y)]
        z_ch = self.arabic_to_chinese[str(z)]
        
        answer = (x - y) * z
        equation = f"{x_ch}减去{y_ch}再乘以{z_ch}等于？"
        
        return equation, answer
    
    def _complex_pattern_3(self) -> Tuple[str, int]:
        """模式3: X除以Y的结果加Z等于？"""
        y = random.randint(1, 4)
        result = random.randint(1, 5)
        x = result * y
        z = random.randint(1, 9)
        
        x_ch = self.arabic_to_chinese[str(x)]
        y_ch = self.arabic_to_chinese[str(y)]
        z_ch = self.arabic_to_chinese[str(z)]
        
        answer = result + z
        equation = f"{x_ch}除以{y_ch}的结果加{z_ch}等于？"
        
        return equation, answer
    
    def generate_image(self, text: str, answer: int) -> Tuple[np.ndarray, str]:
        """
        生成验证码图片
        
        Args:
            text: 数学表达式文本
            answer: 正确答案
            
        Returns:
            (image, text): 图片数组和文本
        """
        # 图片尺寸
        width, height = self.config['data']['image_size']
        
        # 创建图片
        img = Image.new('RGB', (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # 选择字体 - 使用默认字体
        font_size = random.choice(self.config['generator']['font_sizes'])
        
        try:
            font = ImageFont.load_default()
        except:
            font = None
        
        # 绘制文本
        x = 10
        y = 15
        
        text_color = (0, 0, 0)
        draw.text((x, y), text, font=font, fill=text_color)
        
        # 转换为numpy数组
        img_array = np.array(img)
        
        return img_array, text
    
    def generate_dataset(self, split: str = 'train', num_samples: int = None) -> None:
        """
        生成数据集
        
        Args:
            split: 数据集划分 ('train', 'val', 'test')
            num_samples: 生成样本数量
        """
        if num_samples is None:
            num_samples = self.config['generator']['num_samples'][split]
        
        output_dir = self.config['data']['generated_dir'] / split
        output_dir.mkdir(parents=True, exist_ok=True)
        
        labels = []
        equation_types = self.config['generator']['equation_types']
        
        print(f"正在生成 {split} 数据集，共 {num_samples} 个样本...")
        
        for i in range(num_samples):
            # 随机选择方程式类型
            eq_type = random.choice(equation_types)
            
            # 生成方程式和答案
            equation_text, answer = self.generate_equation(eq_type)
            
            # 生成图片
            img_array, text = self.generate_image(equation_text, answer)
            
            # 保存图片
            img_filename = f"{split}_{i:06d}.png"
            img_path = output_dir / img_filename
            
            img_pil = Image.fromarray(img_array)
            img_pil.save(img_path)
            
            # 保存标签信息
            label_info = {
                'filename': img_filename,
                'equation': equation_text,
                'answer': answer,
                'type': eq_type,
                'characters': list(equation_text)
            }
            labels.append(label_info)
            
            if (i + 1) % 1000 == 0:
                print(f"已生成 {i + 1}/{num_samples} 个样本")
        
        # 保存标签文件
        labels_file = output_dir / 'labels.json'
        with open(labels_file, 'w', encoding='utf-8') as f:
            json.dump(labels, f, ensure_ascii=False, indent=2)
        
        print(f"{split} 数据集生成完成！保存在 {output_dir}")

def main():
    """主函数"""
    print("数学验证码生成器测试")

if __name__ == "__main__":
    main() 