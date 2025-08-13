# Qwen3模型训练器使用指南

这是一个专门为Qwen3Transformer模型设计的训练器，包含完整的数据处理、训练和推理功能。

## 🚀 主要特性

- **专用训练器**: 专门为Qwen3Transformer模型优化
- **智能数据处理**: 自动处理变长序列和填充
- **内置分词器**: 简单的字符级分词器，支持中文
- **文本生成**: 训练后可直接进行文本生成
- **自动保存**: 支持检查点保存和恢复
- **梯度裁剪**: 防止梯度爆炸，稳定训练

## 📁 文件结构

```
trainer/
├── qwen3_trainer.py      # Qwen3专用训练器
├── qwen3_example.py      # 完整使用示例
├── trainer.py            # 通用训练器
└── QWEN3_README.md      # 本说明文档
```

## 🔧 安装依赖

确保已安装以下依赖：

```bash
pip install torch>=2.8.0
```

## 📖 快速开始

### 1. 基本使用

```python
from qwen3_trainer import Qwen3Trainer, SimpleTokenizer, TextDataset
from nano_models.model.qwen3 import Qwen3Config, Qwen3Transformer

# 创建模型配置
config = Qwen3Config.from_preset('tiny')  # 可选: tiny, small, medium, large
model = Qwen3Transformer(config)

# 创建分词器和数据集
tokenizer = SimpleTokenizer(vocab_size=1000)
texts = ["你的训练文本1", "你的训练文本2", ...]
dataset = TextDataset(texts, tokenizer, max_length=512)

# 创建训练器
trainer = Qwen3Trainer(
    model=model,
    dataset=dataset,
    tokenizer=tokenizer,
    optimizer=optimizer,
    criterion=criterion,
    batch_size=4
)

# 开始训练
trainer.train(num_epochs=10)
```

### 2. 模型配置选项

```python
# 使用预设配置
config = Qwen3Config.from_preset('tiny')    # 约1.5M参数
config = Qwen3Config.from_preset('small')   # 约6M参数
config = Qwen3Config.from_preset('medium')  # 约24M参数
config = Qwen3Config.from_preset('large')   # 约96M参数

# 自定义配置
config = Qwen3Config(
    hidden_size=512,
    vocab_size=32000,
    intermediate_size=2048,
    num_decoder_layer=12,
    head_dim=64,
    num_attention_heads=8,
    num_key_value_heads=8,
    num_key_value_groups=1,
    attention_bias=False
)
```

### 3. 训练参数调优

```python
# 创建优化器
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,           # 学习率
    weight_decay=0.01   # 权重衰减
)

# 创建损失函数
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# 训练器配置
trainer = Qwen3Trainer(
    model=model,
    dataset=dataset,
    tokenizer=tokenizer,
    optimizer=optimizer,
    criterion=criterion,
    batch_size=4,           # 批次大小
    device="auto",          # 自动选择设备
    save_dir="./checkpoints", # 保存目录
    max_length=512          # 最大序列长度
)
```

## 📊 数据处理

### 1. 文本数据集

```python
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 自动构建词汇表
        self.tokenizer.build_vocab(texts)
        
        # 预处理文本
        self.processed_data = []
        for text in texts:
            tokens = self.tokenizer.encode(text, max_length)
            if len(tokens) >= 2:
                self.processed_data.append(tokens)
    
    def __getitem__(self, idx):
        tokens = self.processed_data[idx]
        # 输入: 除了最后一个token的所有token
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        # 目标: 除了第一个token的所有token
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        return input_ids, target_ids
```

### 2. 分词器功能

```python
tokenizer = SimpleTokenizer(vocab_size=1000)

# 构建词汇表
tokenizer.build_vocab(texts, min_freq=2)

# 编码文本
tokens = tokenizer.encode("你好世界", max_length=64)

# 解码token
text = tokenizer.decode(tokens)
```

## 🎯 训练过程

### 1. 开始训练

```python
# 基本训练
trainer.train(num_epochs=10)

# 高级训练配置
trainer.train(
    num_epochs=20,      # 训练轮数
    save_every=5,       # 每5轮保存一次
    log_every=10        # 每10步记录一次
)
```

### 2. 训练监控

训练过程中会显示：
- 每个epoch的进度
- 每个batch的损失
- 平均损失
- 自动保存信息

### 3. 训练特性

- **梯度裁剪**: 自动防止梯度爆炸
- **自动保存**: 定期保存检查点和最佳模型
- **损失计算**: 忽略填充token的损失
- **设备管理**: 自动处理CPU/GPU数据转移

## 🔍 模型评估

```python
# 评估训练数据集
eval_loss = trainer.eval()

# 评估自定义数据集
eval_dataset = TextDataset(eval_texts, tokenizer, max_length=512)
eval_loss = trainer.eval(eval_dataset)
```

## ✨ 文本生成

训练完成后，可以直接进行文本生成：

```python
# 生成文本
prompt = "人工智能"
generated = trainer.generate_text(
    prompt=prompt,
    max_length=100,    # 最大生成长度
    temperature=0.8    # 温度参数，控制随机性
)

print(f"提示: {prompt}")
print(f"生成: {generated}")
```

## 💾 模型保存和加载

### 1. 自动保存

训练过程中会自动保存：
- `qwen3_checkpoint_epoch_X.pt`: 定期检查点
- `qwen3_best_model.pt`: 最佳模型

### 2. 手动保存

```python
trainer.save("my_model.pt")
```

### 3. 加载模型

```python
# 加载检查点
success = trainer.load("qwen3_checkpoints/qwen3_best_model.pt")

if success:
    print("模型加载成功！")
    # 继续训练或进行推理
```

## 📈 训练统计

```python
# 获取训练统计信息
stats = trainer.get_training_stats()

print(f"当前轮数: {stats['current_epoch']}")
print(f"最佳损失: {stats['best_loss']}")
print(f"总训练步数: {stats['total_training_steps']}")
print(f"使用设备: {stats['device']}")
print(f"词汇表大小: {stats['vocab_size']}")
print(f"最大序列长度: {stats['max_length']}")
```

## 🎨 高级功能

### 1. 自定义损失函数

```python
class CustomLoss(nn.Module):
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=0)
    
    def forward(self, outputs, targets):
        ce_loss = self.ce_loss(outputs, targets)
        # 添加自定义损失项
        return ce_loss
```

### 2. 学习率调度

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=100)

# 在训练循环中
for epoch in range(num_epochs):
    # 训练代码...
    scheduler.step()
```

### 3. 混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# 在训练循环中
with autocast():
    outputs = model(input_ids)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## 🚨 注意事项

1. **内存管理**: 根据GPU内存调整batch_size和max_length
2. **数据质量**: 确保训练文本质量，避免重复和无意义内容
3. **词汇表大小**: 根据实际需求调整vocab_size
4. **序列长度**: 根据任务需求设置合适的max_length
5. **梯度裁剪**: 默认max_norm=1.0，可根据需要调整

## 🔧 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减小batch_size
   - 减小max_length
   - 使用更小的模型配置

2. **训练损失不下降**
   - 检查学习率设置
   - 检查数据质量
   - 增加训练轮数

3. **生成文本质量差**
   - 增加训练数据
   - 调整temperature参数
   - 检查词汇表构建

### 调试技巧

```python
# 检查模型参数
print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

# 检查数据格式
sample = dataset[0]
print(f"输入形状: {sample[0].shape}")
print(f"目标形状: {sample[1].shape}")

# 检查设备
print(f"模型设备: {next(model.parameters()).device}")
print(f"数据设备: {sample[0].device}")
```

## 📚 完整示例

查看 `qwen3_example.py` 文件获取完整的使用示例，包括：
- 数据准备
- 模型创建
- 训练过程
- 文本生成
- 模型加载测试

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个训练器！

## �� 许可证

本项目遵循MIT许可证。
