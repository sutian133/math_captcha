#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CRNN (Convolutional Recurrent Neural Network) 模型
专门用于数学验证码序列识别，支持中文数字、阿拉伯数字和运算符的高精度识别
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CTCLoss
from typing import Dict, Tuple, Optional

class SimpleCNN(nn.Module):
    """简化的CNN特征提取器"""
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            # 第一个卷积块 (160, 60) -> (80, 30)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 第二个卷积块 (80, 30) -> (40, 15)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 第三个卷积块 (40, 15) -> (20, 15)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            
            # 第四个卷积块 (20, 15) -> (10, 15)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
        )
        
        self.feature_dim = 256
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, 3, 60, 160)
        Returns:
            features: (batch_size, feature_dim*H, W)
        """
        x = self.features(x)  # (B, 256, H, W)
        
        # 展平高度维度
        b, c, h, w = x.size()
        # 确保正确计算特征维度
        x = x.view(b, c * h, w)  # (B, 256*H, W)
        
        return x, c * h  # 返回特征和实际的特征维度

class BidirectionalLSTM(nn.Module):
    """双向LSTM层"""
    
    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            bidirectional=True, 
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            output: (batch_size, seq_len, output_size)
        """
        lstm_out, _ = self.lstm(x)  # (B, T, 2*hidden_size)
        output = self.linear(lstm_out)  # (B, T, output_size)
        return output

class CRNN(nn.Module):
    """CRNN模型：CNN特征提取 + RNN序列建模 + CTC损失"""
    
    def __init__(self, num_classes, hidden_size=256):
        super(CRNN, self).__init__()
        
        self.num_classes = num_classes
        
        # CNN特征提取器
        self.cnn = SimpleCNN()
        
        # 计算实际的特征维度
        # 在实际运行时动态计算，这里先设置一个默认值
        self.feature_dim = None
        
        # RNN层 - 延迟初始化
        self.rnn = None
        self.classifier = None
        
        # CTC损失函数
        self.ctc_loss = CTCLoss(blank=num_classes - 1, reduction='mean', zero_infinity=True)
    
    def _init_rnn_if_needed(self, feature_dim):
        """根据实际特征维度初始化RNN"""
        if self.rnn is None:
            self.rnn = BidirectionalLSTM(
                input_size=feature_dim,
                hidden_size=256,
                output_size=256
            )
            self.classifier = nn.Linear(256, self.num_classes)
            
            # 移动到正确的设备
            if next(self.parameters()).is_cuda:
                self.rnn = self.rnn.cuda()
                self.classifier = self.classifier.cuda()
    
    def forward(self, images):
        """
        前向传播
        
        Args:
            images: (batch_size, 3, 60, 160)
            
        Returns:
            log_probs: (seq_len, batch_size, num_classes) - CTC格式
        """
        # CNN特征提取
        cnn_features, feature_dim = self.cnn(images)  # (B, feature_dim*H, W)
        
        # 根据实际特征维度初始化RNN
        self._init_rnn_if_needed(feature_dim)
        
        # 转换维度为RNN输入格式
        cnn_features = cnn_features.permute(0, 2, 1)  # (B, W, feature_dim*H)
        
        # RNN处理
        rnn_output = self.rnn(cnn_features)  # (B, W, hidden_size)
        
        # 分类
        logits = self.classifier(rnn_output)  # (B, W, num_classes)
        
        # 转换为CTC格式 (seq_len, batch_size, num_classes)
        log_probs = F.log_softmax(logits, dim=2)
        log_probs = log_probs.permute(1, 0, 2)  # (W, B, num_classes)
        
        return log_probs
    
    def compute_loss(self, log_probs, targets, input_lengths, target_lengths):
        """计算CTC损失"""
        return self.ctc_loss(log_probs, targets, input_lengths, target_lengths)

class MathCaptchaModel(nn.Module):
    """数学验证码识别的完整模型封装"""
    
    def __init__(self, config: Dict):
        super(MathCaptchaModel, self).__init__()
        
        self.config = config
        self.char_to_idx = config['characters']['char_to_idx']
        self.idx_to_char = config['characters']['idx_to_char']
        self.num_classes = config['characters']['num_classes']
        
        # 核心CRNN模型
        self.crnn = CRNN(self.num_classes)
    
    def forward(self, images):
        """前向传播"""
        return self.crnn(images)
    
    def compute_loss(self, log_probs, targets, input_lengths, target_lengths):
        """计算损失"""
        return self.crnn.compute_loss(log_probs, targets, input_lengths, target_lengths)
    
    def predict(self, images):
        """预测数学表达式"""
        self.eval()
        with torch.no_grad():
            log_probs = self.forward(images)
            seq_len, batch_size = log_probs.size(0), log_probs.size(1)
            
            # 简单贪婪解码
            _, max_indices = torch.max(log_probs, dim=2)  # (seq_len, batch_size)
            max_indices = max_indices.transpose(0, 1)  # (batch_size, seq_len)
            
            results = []
            blank_idx = self.num_classes - 1
            
            for i in range(batch_size):
                sequence = max_indices[i].tolist()
                
                # 移除空白符和重复字符
                decoded_seq = []
                prev_char = None
                for char in sequence:
                    if char != blank_idx and char != prev_char:
                        decoded_seq.append(char)
                    prev_char = char
                
                # 转换为文本
                text = ''.join([self.idx_to_char.get(idx, '') for idx in decoded_seq])
                results.append({'text': text, 'answer': None})
            
            return results

def create_model(config: Dict) -> MathCaptchaModel:
    """创建数学验证码识别模型"""
    return MathCaptchaModel(config)

if __name__ == "__main__":
    print("CRNN模型测试") 