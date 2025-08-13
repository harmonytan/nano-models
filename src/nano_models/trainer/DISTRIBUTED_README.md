# Qwen3分布式训练指南

这个指南将帮助你使用分布式训练来加速Qwen3模型的训练过程。

## 🚀 分布式训练的优势

### 1. **训练加速**
- 多GPU并行训练，显著减少训练时间
- 支持更大的批次大小，提高训练效率
- 更好的梯度估计，可能提高收敛速度

### 2. **内存扩展**
- 每个GPU处理部分数据，减少单GPU内存压力
- 支持更大的模型和更长的序列

### 3. **成本效益**
- 充分利用多GPU资源
- 减少单次训练的时间成本

## 📁 文件结构

```
trainer/
├── distributed_trainer.py      # 分布式训练器核心
├── run_distributed_training.py # 分布式训练启动脚本
├── qwen3_trainer.py           # 单GPU训练器
└── DISTRIBUTED_README.md      # 本说明文档
```

## 🔧 环境要求

### 1. **硬件要求**
- 多GPU环境（推荐2-8张GPU）
- 每张GPU至少8GB显存（推荐16GB+）
- GPU之间支持NCCL通信

### 2. **软件要求**
```bash
# PyTorch版本要求
torch >= 2.0.0
torchvision >= 0.15.0

# 确保支持CUDA和分布式训练
python -c "import torch; print(torch.cuda.is_available()); print(torch.distributed.is_available())"
```

### 3. **网络要求**
- 单机多卡：无需额外网络配置
- 多机多卡：需要配置网络通信

## 🎯 快速开始

### 1. **单机多卡训练**

```bash
# 使用2张GPU训练
python -m torch.distributed.launch --nproc_per_node=2 run_distributed_training.py

# 使用4张GPU训练
python -m torch.distributed.launch --nproc_per_node=4 run_distributed_training.py

# 使用所有可用GPU
python -m torch.distributed.launch --nproc_per_node=8 run_distributed_training.py
```

### 2. **自定义参数训练**

```bash
python -m torch.distributed.launch --nproc_per_node=4 run_distributed_training.py \
    --batch_size 4 \
    --max_length 256 \
    --num_epochs 10 \
    --model_size medium \
    --lr 0.0005 \
    --save_dir ./my_checkpoints
```

### 3. **单GPU训练（兼容性）**

```bash
# 直接运行，自动检测为单GPU模式
python run_distributed_training.py --batch_size 8 --num_epochs 5
```

## 📊 参数说明

### 1. **训练参数**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--local_rank` | int | -1 | 本地GPU排名（自动设置） |
| `--world_size` | int | -1 | 总GPU数量（自动设置） |
| `--batch_size` | int | 2 | 每个GPU的批次大小 |
| `--max_length` | int | 128 | 最大序列长度 |
| `--num_epochs` | int | 3 | 训练轮数 |
| `--model_size` | str | tiny | 模型大小（tiny/small/medium/large） |
| `--lr` | float | 0.001 | 学习率 |
| `--save_dir` | str | ./qwen3_distributed_checkpoints | 保存目录 |

### 2. **模型大小配置**

| 模型大小 | 参数数量 | 显存需求 | 适用场景 |
|----------|----------|----------|----------|
| `tiny` | ~22M | 2-4GB | 快速测试、原型开发 |
| `small` | ~6M | 4-8GB | 小规模任务、资源受限 |
| `medium` | ~24M | 8-16GB | 中等规模任务 |
| `large` | ~96M | 16GB+ | 大规模任务、生产环境 |

## 🔄 分布式训练流程

### 1. **环境初始化**
```python
# 自动检测分布式环境
if args.local_rank != -1:
    setup_distributed_training(args.local_rank, args.world_size)
    is_distributed = True
else:
    is_distributed = False
```

### 2. **模型包装**
```python
# 分布式训练时自动包装为DDP
if self.is_distributed:
    self.model = DDP(self.model, device_ids=[local_rank], output_device=local_rank)
```

### 3. **数据分发**
```python
# 使用分布式采样器
if self.is_distributed:
    sampler = DistributedSampler(
        dataset, 
        num_replicas=self.world_size, 
        rank=self.rank,
        shuffle=True
    )
```

### 4. **梯度同步**
```python
# 自动梯度同步（DDP处理）
loss.backward()
self.optimizer.step()
```

### 5. **损失同步**
```python
# 在分布式环境中同步损失
if self.is_distributed:
    avg_loss_tensor = torch.tensor(avg_loss, device=self.device)
    dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
    avg_loss = avg_loss_tensor.item() / self.world_size
```

## 📈 性能优化技巧

### 1. **批次大小调优**
```bash
# 根据GPU显存调整批次大小
# 显存充足时增加批次大小
--batch_size 8  # 每GPU 8个样本

# 显存不足时减少批次大小
--batch_size 1  # 每GPU 1个样本
```

### 2. **序列长度优化**
```bash
# 根据任务需求调整序列长度
--max_length 64   # 短文本任务
--max_length 512  # 长文本任务
--max_length 1024 # 超长文本任务
```

### 3. **学习率调整**
```bash
# 分布式训练时可能需要调整学习率
--lr 0.001    # 默认学习率
--lr 0.0005   # 降低学习率
--lr 0.002    # 提高学习率
```

### 4. **数据加载优化**
```python
# 多进程数据加载
DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=sampler,
    num_workers=4,      # 多进程
    pin_memory=True     # 内存固定
)
```

## 🚨 常见问题和解决方案

### 1. **CUDA内存不足**
```bash
# 解决方案1：减少批次大小
--batch_size 1

# 解决方案2：减少序列长度
--max_length 64

# 解决方案3：使用更小的模型
--model_size tiny
```

### 2. **NCCL通信错误**
```bash
# 解决方案1：设置环境变量
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

# 解决方案2：使用GLOO后端（CPU训练）
# 修改 distributed_trainer.py 中的后端设置
```

### 3. **进程同步问题**
```bash
# 解决方案：确保所有进程同时启动
# 使用 torch.distributed.launch 自动处理
python -m torch.distributed.launch --nproc_per_node=N script.py
```

### 4. **模型保存失败**
```python
# 解决方案：只在主进程中保存
if self.rank == 0:
    self.save("model.pt")
```

## 🔍 监控和调试

### 1. **训练监控**
```python
# 查看训练统计
stats = trainer.get_training_stats()
print(f"进程排名: {stats['rank']}")
print(f"总进程数: {stats['world_size']}")
print(f"是否分布式: {stats['is_distributed']}")
```

### 2. **性能分析**
```bash
# 使用nvidia-smi监控GPU使用
watch -n 1 nvidia-smi

# 使用torch profiler分析性能
python -m torch.utils.bottleneck script.py
```

### 3. **日志分析**
```python
# 分布式训练日志示例
# 进程 0/4 在 GPU 0 上初始化完成
# 进程 1/4 在 GPU 1 上初始化完成
# 进程 2/4 在 GPU 2 上初始化完成
# 进程 3/4 在 GPU 3 上初始化完成
# 分布式Qwen3训练器初始化完成
# 总批次大小: 8  # 2 * 4
```

## 🚀 高级用法

### 1. **多机多卡训练**
```bash
# 机器1（主节点）
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="192.168.1.100" \
    --master_port=12355 \
    run_distributed_training.py

# 机器2（从节点）
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr="192.168.1.100" \
    --master_port=12355 \
    run_distributed_training.py
```

### 2. **混合精度训练**
```python
# 在训练循环中添加
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = self.model(input_ids)
    loss = self.criterion(outputs_flat, targets_flat)

scaler.scale(loss).backward()
scaler.step(self.optimizer)
scaler.update()
```

### 3. **梯度累积**
```python
# 在训练循环中添加
accumulation_steps = 4

if (batch_idx + 1) % accumulation_steps == 0:
    self.optimizer.step()
    self.optimizer.zero_grad()
```

## 📚 完整示例

### 1. **快速测试**
```bash
# 使用2张GPU快速测试
python -m torch.distributed.launch --nproc_per_node=2 run_distributed_training.py \
    --model_size tiny \
    --num_epochs 1 \
    --batch_size 1
```

### 2. **生产训练**
```bash
# 使用4张GPU进行生产训练
python -m torch.distributed.launch --nproc_per_node=4 run_distributed_training.py \
    --model_size medium \
    --num_epochs 100 \
    --batch_size 4 \
    --max_length 512 \
    --lr 0.0005 \
    --save_dir ./production_checkpoints
```

### 3. **大规模训练**
```bash
# 使用8张GPU进行大规模训练
python -m torch.distributed.launch --nproc_per_node=8 run_distributed_training.py \
    --model_size large \
    --num_epochs 200 \
    --batch_size 2 \
    --max_length 1024 \
    --lr 0.0001
```

## 🤝 故障排除

### 1. **检查分布式环境**
```python
import torch.distributed as dist

print(f"分布式是否初始化: {dist.is_initialized()}")
if dist.is_initialized():
    print(f"当前排名: {dist.get_rank()}")
    print(f"总进程数: {dist.get_world_size()}")
```

### 2. **检查GPU状态**
```bash
# 检查GPU数量和状态
nvidia-smi

# 检查CUDA版本
python -c "import torch; print(torch.version.cuda)"
```

### 3. **检查网络配置**
```bash
# 检查端口是否被占用
netstat -an | grep 12355

# 检查防火墙设置
sudo ufw status
```

## 📖 参考资料

- [PyTorch分布式训练官方文档](https://pytorch.org/docs/stable/distributed.html)
- [NCCL后端文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- [分布式训练最佳实践](https://pytorch.org/tutorials/beginner/dist_overview.html)

## 🎉 总结

使用这个分布式训练器，你可以：

1. **轻松扩展到多GPU环境**
2. **显著加速训练过程**
3. **处理更大的模型和数据**
4. **保持与单GPU训练的兼容性**

记住：分布式训练需要更多的配置和调试，但一旦配置正确，训练效率将大幅提升！
