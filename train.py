#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复版本2：解决序列长度不匹配问题
核心改进：输入序列从10增加到40，BLANK索引调整
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_dataset_chars(data_dir):
    labels_file = data_dir / 'labels.json'
    with open(labels_file, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    char_counter = Counter()
    max_length = 0
    for sample in samples:
        for char in sample['characters']:
            char_counter[char] += 1
        max_length = max(max_length, len(sample['characters']))
    
    logger.info(f"数据集字符分析 (共{len(char_counter)}个不同字符):")
    for char, count in char_counter.most_common(15):
        logger.info(f"  '{char}': {count}")
    logger.info(f"最大序列长度: {max_length}")
    
    return list(char_counter.keys()), max_length

def create_char_mapping(dataset_chars):
    # BLANK放在最后
    all_chars = sorted(dataset_chars) + ['<BLANK>']
    char_to_idx = {char: idx for idx, char in enumerate(all_chars)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    
    logger.info(f"字符映射: {len(all_chars)}个字符, BLANK索引: {char_to_idx['<BLANK>']}")
    return char_to_idx, idx_to_char

class FixedCNN(nn.Module):
    def __init__(self):
        super(FixedCNN, self).__init__()
        
        self.features = nn.Sequential(
            # (3, 60, 160)
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 2)),  # (64, 30, 80)
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 2)),  # (128, 15, 40)
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d((1, 2)),  # (256, 15, 20)
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),  # (256, 15, 20)
        )
        self.feature_dim = 256 * 15  # 3840
    
    def forward(self, x):
        x = self.features(x)  # (B, 256, 15, 20)
        b, c, h, w = x.size()
        x = x.view(b, c * h, w).permute(0, 2, 1)  # (B, 20, 3840)
        
        # 上采样到40个时间步 (关键修复!)
        x = x.permute(0, 2, 1)  # (B, 3840, 20)
        x = torch.nn.functional.interpolate(x, size=40, mode='linear', align_corners=False)
        x = x.permute(0, 2, 1)  # (B, 40, 3840)
        
        return x

class FixedCRNN(nn.Module):
    def __init__(self, num_classes):
        super(FixedCRNN, self).__init__()
        self.num_classes = num_classes
        self.blank_idx = num_classes - 1  # BLANK在最后
        
        self.cnn = FixedCNN()
        self.lstm = nn.LSTM(self.cnn.feature_dim, 256, 1, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(512, num_classes)
        self.ctc_loss = nn.CTCLoss(blank=self.blank_idx, reduction='mean', zero_infinity=True)
    
    def forward(self, images):
        cnn_features = self.cnn(images)  # (B, 40, 3840)
        lstm_out, _ = self.lstm(cnn_features)  # (B, 40, 512)
        logits = self.classifier(lstm_out)  # (B, 40, num_classes)
        log_probs = torch.log_softmax(logits, dim=2)
        return log_probs.permute(1, 0, 2)  # (40, B, num_classes)
    
    def compute_loss(self, log_probs, targets, input_lengths, target_lengths):
        return self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
    
    def decode(self, log_probs, idx_to_char):
        _, max_indices = torch.max(log_probs, dim=2)
        max_indices = max_indices.transpose(0, 1)
        
        results = []
        for i in range(max_indices.size(0)):
            sequence = max_indices[i].tolist()
            decoded = []
            prev_char = None
            for char_idx in sequence:
                if char_idx != self.blank_idx and char_idx != prev_char:
                    if char_idx in idx_to_char:
                        decoded.append(idx_to_char[char_idx])
                prev_char = char_idx
            results.append(''.join(decoded))
        return results

class MathCaptchaDataset(Dataset):
    def __init__(self, data_dir, char_to_idx, split='train'):
        self.data_dir = Path(data_dir)
        self.char_to_idx = char_to_idx
        self.split = split
        
        labels_file = self.data_dir / 'labels.json'
        with open(labels_file, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)
        
        logger.info(f"加载{split}数据集: {len(self.samples)}个样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        img_path = self.data_dir / sample['filename']
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, (160, 60))
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        
        text = sample['equation']
        char_indices = []
        for char in text:
            if char in self.char_to_idx:
                char_indices.append(self.char_to_idx[char])
        
        return img, char_indices, text, sample['answer']

def collate_fn(batch):
    images, char_indices_list, texts, answers = zip(*batch)
    images = torch.stack(images, 0)
    
    batch_size = len(char_indices_list)
    input_lengths = torch.full((batch_size,), 40, dtype=torch.long)  # 修复：40时间步
    
    targets = []
    target_lengths = []
    for char_indices in char_indices_list:
        targets.extend(char_indices)
        target_lengths.append(len(char_indices))
    
    targets = torch.tensor(targets, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    
    return images, targets, input_lengths, target_lengths, texts, answers

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    train_data_dir = Path('data/generated/train')
    test_data_dir = Path('data/generated/test')
    
    dataset_chars, max_length = analyze_dataset_chars(train_data_dir)
    char_to_idx, idx_to_char = create_char_mapping(dataset_chars)
    
    logger.info(f"🔧 关键修复: 输入40步 vs 目标{max_length}步 = {40/max_length:.1f}倍 (之前是{10/max_length:.1f}倍)")
    
    train_dataset = MathCaptchaDataset(train_data_dir, char_to_idx, 'train')
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=0)
    
    test_loader = None
    if test_data_dir.exists():
        test_dataset = MathCaptchaDataset(test_data_dir, char_to_idx, 'test')
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    model = FixedCRNN(num_classes=len(char_to_idx)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    best_loss = float('inf')
    for epoch in range(50):
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/50')
        for images, targets, input_lengths, target_lengths, texts, answers in pbar:
            images = images.to(device)
            targets = targets.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)
            
            optimizer.zero_grad()
            log_probs = model(images)
            loss = model.compute_loss(log_probs, targets, input_lengths, target_lengths)
            
            if not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                train_loss += loss.item()
                train_batches += 1
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / train_batches if train_batches > 0 else float('inf')
        logger.info(f'Epoch {epoch+1}: 训练损失 = {avg_train_loss:.4f}')
        
        if test_loader:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for images, targets, input_lengths, target_lengths, texts, answers in test_loader:
                    images = images.to(device)
                    targets = targets.to(device)
                    input_lengths = input_lengths.to(device)
                    target_lengths = target_lengths.to(device)
                    
                    log_probs = model(images)
                    loss = model.compute_loss(log_probs, targets, input_lengths, target_lengths)
                    
                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        val_loss += loss.item()
                        val_batches += 1
                    
                    predictions = model.decode(log_probs, idx_to_char)
                    for pred, truth in zip(predictions, texts):
                        if pred == truth:
                            correct += 1
                        total += 1
            
            avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
            accuracy = correct / total * 100 if total > 0 else 0
            
            logger.info(f'Epoch {epoch+1}: 验证损失 = {avg_val_loss:.4f}, 准确率 = {accuracy:.2f}%')
            scheduler.step(avg_val_loss)
            
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'char_to_idx': char_to_idx,
                    'idx_to_char': idx_to_char,
                    'epoch': epoch,
                    'loss': best_loss,
                    'accuracy': accuracy
                }, checkpoint_dir / 'best_model_v2.pth')
                logger.info(f'✅ 保存最佳模型 (损失: {best_loss:.4f}, 准确率: {accuracy:.2f}%)')
    
    logger.info('训练完成!')

if __name__ == "__main__":
    train_model() 