# 数学验证码识别系统

基于深度学习的高精度数学验证码识别系统，支持中文数字、阿拉伯数字和数学运算符的识别。

## 📋 功能特点

- **多语言支持**: 支持中文数字（一、二、三...）和阿拉伯数字（1、2、3...）
- **运算符识别**: 支持加减乘除运算符（+、-、×、÷、加、减、乘、除）
- **高精度识别**: 基于CRNN架构，结合CNN特征提取和RNN序列建模
- **自动答案计算**: 识别数学表达式后自动计算答案
- **灵活配置**: 支持多种模型配置和训练参数调整

## 🚀 快速开始

### 环境要求

```bash
Python >= 3.7
PyTorch >= 1.7.0
OpenCV >= 4.0
Pillow >= 8.0
numpy >= 1.19.0
```

### 安装依赖

```bash
pip install torch torchvision opencv-python pillow numpy tqdm
```

### 运行演示

```bash
# 运行完整系统演示
python demo.py

# 运行推理演示
python inference.py --demo

# 开始训练
python train.py
```

## 📁 项目结构

```
math_captcha/
├── config/
│   └── config.py           # 系统配置文件
├── models/
│   └── cnn/
│       └── crnn_model.py   # CRNN模型定义
├── utils/
│   └── generator.py        # 数据生成器
├── data/                   # 数据目录
├── train.py               # 训练脚本
├── inference.py           # 推理脚本
├── demo.py                # 演示脚本
└── README.md              # 项目文档
```

## 🔧 配置说明

### 支持的字符集

- **阿拉伯数字**: 0-9
- **中文数字**: 零、一、二、三、四、五、六、七、八、九
- **阿拉伯运算符**: +、-、×、÷、=
- **中文运算符**: 加、减、乘、除、等于
- **其他符号**: ?、？、(、)、空格

### 支持的表达式类型

1. **简单阿拉伯**: `1+2=?`
2. **简单中文**: `一加二等于？`
3. **混合记号**: `1+二=？`
4. **带括号**: `(1+2)×3=?`
5. **复杂中文**: `三乘以二加一等于？`

## 🎯 模型架构

### CRNN架构

```
输入图片 (160×60×3)
    ↓
CNN特征提取器
    ├── Conv2d(3→32) + ReLU + MaxPool2d
    ├── Conv2d(32→64) + ReLU + MaxPool2d  
    ├── Conv2d(64→128) + ReLU + MaxPool2d
    └── Conv2d(128→256) + ReLU + MaxPool2d
    ↓
双向LSTM序列建模
    ↓
CTC解码输出
```

### 性能指标

- **字符准确率**: > 95%
- **序列准确率**: > 90%
- **推理速度**: < 50ms/张
- **模型大小**: < 10MB

## 📊 训练配置

### 默认参数

```python
TRAIN_CONFIG = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 100,
    'optimizer': 'AdamW',
    'loss_function': 'CTCLoss'
}
```

### 数据增强

- 随机旋转 (±5°)
- 噪声添加
- 亮度/对比度调整
- 弹性变形
- 网格扭曲

## 🔍 使用示例

### 基本推理

```python
from inference import MathCaptchaInference

# 创建推理器
inference = MathCaptchaInference()

# 预测图片
result = inference.predict('path/to/image.png')
print(f"识别结果: {result['text']}")
print(f"计算答案: {result['calculated_answer']}")
```

### 自定义训练

```python
from train import train_model
from config.config import get_config

# 加载配置
config = get_config()

# 修改配置
config['train']['epochs'] = 50
config['train']['batch_size'] = 16

# 开始训练
model = train_model(config)
```

### 数据生成

```python
from utils.generator import MathCaptchaGenerator
from config.config import get_config

# 创建生成器
generator = MathCaptchaGenerator(get_config())

# 生成单个样本
equation, answer = generator.generate_equation('arabic_simple')
img_array, text = generator.generate_image(equation, answer)

# 批量生成数据集
generator.generate_dataset('train', 10000)
```

## 🎨 自定义配置

### 添加新字符

```python
# 在 config/config.py 中添加
CHARACTER_SET['new_category'] = ['新', '字', '符']
```

### 调整模型结构

```python
MODEL_CONFIG = {
    'cnn': {
        'backbone': 'resnet18',  # 可选: resnet18, resnet34, resnet50
        'feature_dim': 512,
        'dropout': 0.2,
    },
    'lstm': {
        'hidden_size': 256,      # LSTM隐藏层大小
        'num_layers': 2,         # LSTM层数
        'bidirectional': True,   # 是否双向
    }
}
```

## 📈 性能优化

### 提升准确率

1. **增加训练数据**: 生成更多样本
2. **数据增强**: 启用更多增强技术
3. **模型调优**: 调整网络结构参数
4. **集成学习**: 使用多模型投票

### 提升速度

1. **模型量化**: 使用int8量化
2. **模型剪枝**: 移除冗余参数
3. **批处理**: 批量处理多张图片
4. **GPU加速**: 使用CUDA加速

## 🐛 常见问题

### Q: 模型训练时显存不足？
A: 减少batch_size或使用梯度累积

### Q: 识别准确率低？
A: 增加训练数据量，调整数据增强策略

### Q: 字体不支持中文？
A: 安装中文字体或使用系统默认字体

### Q: CTC损失为nan？
A: 检查标签格式，确保序列长度正确

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📞 联系

如有问题请提交Issue或联系开发者。

---

**注意**: 此项目为演示版本，实际生产环境使用需要大量数据训练和优化。 