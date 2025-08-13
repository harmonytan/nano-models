"""
支持分布式训练的Qwen3训练器
"""
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.distributed import DistributedSampler
from typing import Optional, Dict, Any, List, Union
import os
import json
from datetime import datetime
import collections
from collections import Counter

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from nano_models.model.qwen3 import Qwen3Config, Qwen3Transformer


class SimpleTokenizer:
    """
    简单的分词器，用于演示目的
    在实际使用中，建议使用更专业的分词器如sentencepiece或tiktoken
    """
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        
        # 简单的词汇表映射
        self.token_to_id = {
            '<pad>': self.pad_token_id,
            '<unk>': self.unk_token_id,
            '<bos>': self.bos_token_id,
            '<eos>': self.eos_token_id,
        }
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        
        # 字符级别的简单分词
        self.char_vocab = set()
    
    def build_vocab(self, texts: List[str], min_freq: int = 2):
        """从文本构建词汇表"""
        # 统计字符频率
        char_counter = Counter()
        for text in texts:
            char_counter.update(text)
        
        # 选择频率足够的字符
        for char, freq in char_counter.items():
            if freq >= min_freq and len(self.token_to_id) < self.vocab_size:
                self.token_to_id[char] = len(self.token_to_id)
                self.id_to_token[len(self.token_to_id) - 1] = char
                self.char_vocab.add(char)
        
        if dist.is_initialized() and dist.get_rank() == 0:
            print(f"词汇表构建完成，大小: {len(self.token_to_id)}")
    
    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """将文本编码为token ID"""
        tokens = []
        
        # 添加开始标记
        tokens.append(self.bos_token_id)
        
        # 字符级分词
        for char in text:
            if char in self.token_to_id:
                tokens.append(self.token_to_id[char])
            else:
                tokens.append(self.unk_token_id)
        
        # 添加结束标记
        tokens.append(self.eos_token_id)
        
        # 截断或填充
        if max_length:
            if len(tokens) > max_length:
                tokens = tokens[:max_length-1] + [self.eos_token_id]
            else:
                tokens.extend([self.pad_token_id] * (max_length - len(tokens)))
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """将token ID解码为文本"""
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if token not in ['<pad>', '<bos>', '<eos>']:
                    tokens.append(token)
            else:
                tokens.append('<unk>')
        
        return ''.join(tokens)


class TextDataset:
    """文本数据集，用于训练语言模型"""
    
    def __init__(self, texts: List[str], tokenizer: SimpleTokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 构建词汇表
        self.tokenizer.build_vocab(texts)
        
        # 预处理所有文本
        self.processed_data = []
        for text in texts:
            tokens = self.tokenizer.encode(text, max_length)
            if len(tokens) >= 2:  # 至少需要输入和目标
                self.processed_data.append(tokens)
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        tokens = self.processed_data[idx]
        
        # 输入是除了最后一个token的所有token
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        # 目标是除了第一个token的所有token
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        
        return input_ids, target_ids


class DistributedQwen3Trainer:
    """
    支持分布式训练的Qwen3训练器
    """
    
    def __init__(self, 
                 model: Qwen3Transformer,
                 dataset: TextDataset,
                 tokenizer: SimpleTokenizer,
                 optimizer: torch.optim.Optimizer,
                 criterion: nn.Module,
                 batch_size: int = 4,
                 device: str = "auto",
                 save_dir: str = "./qwen3_distributed_checkpoints",
                 max_length: int = 512,
                 local_rank: int = -1,
                 world_size: int = -1):
        """
        初始化分布式Qwen3训练器
        
        Args:
            model: Qwen3Transformer模型
            dataset: 文本数据集
            tokenizer: 分词器
            optimizer: 优化器
            criterion: 损失函数
            batch_size: 每个GPU的批次大小
            device: 设备 ("auto", "cpu", "cuda")
            save_dir: 保存目录
            max_length: 最大序列长度
            local_rank: 本地GPU排名
            world_size: 总GPU数量
        """
        self.model = model
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_size = batch_size
        self.max_length = max_length
        self.save_dir = save_dir
        self.local_rank = local_rank
        self.world_size = world_size
        
        # 检查是否在分布式环境中
        self.is_distributed = dist.is_initialized()
        
        if self.is_distributed:
            self.device = torch.device(f"cuda:{local_rank}")
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            # 自动选择设备
            if device == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)
            self.rank = 0
            self.world_size = 1
        
        # 将模型移到设备上
        self.model.to(self.device)
        
        # 如果是分布式训练，包装模型
        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[local_rank], output_device=local_rank)
            print(f"进程 {self.rank}/{self.world_size} 在 GPU {local_rank} 上初始化完成")
        else:
            print(f"单GPU训练在设备 {self.device} 上初始化完成")
        
        # 创建数据加载器
        if self.is_distributed:
            # 分布式采样器
            sampler = DistributedSampler(
                dataset, 
                num_replicas=self.world_size, 
                rank=self.rank,
                shuffle=True
            )
            self.dataloader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                sampler=sampler,
                collate_fn=self._collate_fn,
                num_workers=4,  # 多进程数据加载
                pin_memory=True
            )
        else:
            self.dataloader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=True,
                collate_fn=self._collate_fn
            )
        
        # 训练状态
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.training_history = []
        
        # 创建保存目录（只在主进程中）
        if self.rank == 0:
            os.makedirs(save_dir, exist_ok=True)
        
        if self.rank == 0:
            print(f"分布式Qwen3训练器初始化完成")
            print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
            print(f"词汇表大小: {tokenizer.vocab_size}")
            print(f"最大序列长度: {max_length}")
            print(f"总批次大小: {batch_size * self.world_size}")
    
    def _collate_fn(self, batch):
        """自定义数据整理函数，处理变长序列"""
        input_ids, target_ids = zip(*batch)
        
        # 找到最大长度
        max_len = max(len(ids) for ids in input_ids)
        
        # 填充到相同长度
        padded_inputs = []
        padded_targets = []
        
        for input_seq, target_seq in zip(input_ids, target_ids):
            # 填充输入
            if len(input_seq) < max_len:
                padding = [self.tokenizer.pad_token_id] * (max_len - len(input_seq))
                input_seq = torch.cat([input_seq, torch.tensor(padding)])
            padded_inputs.append(input_seq)
            
            # 填充目标
            if len(target_seq) < max_len:
                padding = [self.tokenizer.pad_token_id] * (max_len - len(target_seq))
                target_seq = torch.cat([target_seq, torch.tensor(padding)])
            padded_targets.append(target_seq)
        
        # 堆叠为批次
        batch_inputs = torch.stack(padded_inputs)
        batch_targets = torch.stack(padded_targets)
        
        return batch_inputs, batch_targets
    
    def train(self, num_epochs: int = 10, save_every: int = 5, log_every: int = 10):
        """
        训练模型
        
        Args:
            num_epochs: 训练轮数
            save_every: 每多少轮保存一次
            log_every: 每多少步记录一次
        """
        if self.rank == 0:
            print(f"开始分布式训练Qwen3模型，共 {num_epochs} 轮")
            print(f"总批次大小: {self.batch_size * self.world_size}")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            
            # 设置epoch（分布式采样器需要）
            if self.is_distributed:
                self.dataloader.sampler.set_epoch(epoch)
            
            epoch_loss = 0.0
            num_batches = 0
            
            # 设置为训练模式
            self.model.train()
            
            for batch_idx, (input_ids, target_ids) in enumerate(self.dataloader):
                # 将数据移到设备上
                input_ids = input_ids.to(self.device, non_blocking=True)
                target_ids = target_ids.to(self.device, non_blocking=True)
                
                # 前向传播
                self.optimizer.zero_grad()
                outputs = self.model(input_ids)
                
                # 计算损失 - 语言模型损失
                outputs_flat = outputs.view(-1, outputs.size(-1))
                targets_flat = target_ids.view(-1)
                loss = self.criterion(outputs_flat, targets_flat)
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # 打印进度（只在主进程中）
                if self.rank == 0 and batch_idx % log_every == 0:
                    print(f"Epoch {self.current_epoch}/{num_epochs}, "
                          f"Batch {batch_idx}/{len(self.dataloader)}, "
                          f"Loss: {loss.item():.4f}")
            
            # 计算平均损失
            avg_loss = epoch_loss / num_batches
            
            # 在分布式环境中同步损失
            if self.is_distributed:
                avg_loss_tensor = torch.tensor(avg_loss, device=self.device)
                dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss_tensor.item() / self.world_size
            
            # 记录训练历史（只在主进程中）
            if self.rank == 0:
                self.training_history.append({
                    'epoch': self.current_epoch,
                    'loss': avg_loss,
                    'timestamp': datetime.now().isoformat()
                })
                
                print(f"Epoch {self.current_epoch} 完成，平均损失: {avg_loss:.4f}")
                
                # 保存检查点
                if self.current_epoch % save_every == 0:
                    self.save(f"qwen3_distributed_checkpoint_epoch_{self.current_epoch}.pt")
                
                # 更新最佳损失
                if avg_loss < self.best_loss:
                    self.best_loss = avg_loss
                    self.save("qwen3_distributed_best_model.pt")
        
        if self.rank == 0:
            print("训练完成！")
    
    def eval(self, eval_dataset: Optional[TextDataset] = None):
        """
        评估模型
        
        Args:
            eval_dataset: 评估数据集，如果为None则使用训练数据集
        """
        if eval_dataset is None:
            eval_dataset = self.dataset
        
        # 创建评估数据加载器
        if self.is_distributed:
            eval_sampler = DistributedSampler(
                eval_dataset, 
                num_replicas=self.world_size, 
                rank=self.rank,
                shuffle=False
            )
            eval_dataloader = DataLoader(
                eval_dataset, 
                batch_size=self.batch_size, 
                sampler=eval_sampler,
                collate_fn=self._collate_fn,
                num_workers=4,
                pin_memory=True
            )
        else:
            eval_dataloader = DataLoader(
                eval_dataset, 
                batch_size=self.batch_size, 
                shuffle=False,
                collate_fn=self._collate_fn
            )
        
        # 设置为评估模式
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for input_ids, target_ids in eval_dataloader:
                # 将数据移到设备上
                input_ids = input_ids.to(self.device, non_blocking=True)
                target_ids = target_ids.to(self.device, non_blocking=True)
                
                # 前向传播
                outputs = self.model(input_ids)
                
                # 计算损失
                outputs_flat = outputs.view(-1, outputs.size(-1))
                targets_flat = target_ids.view(-1)
                loss = self.criterion(outputs_flat, targets_flat)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # 在分布式环境中同步损失
        if self.is_distributed:
            avg_loss_tensor = torch.tensor(avg_loss, device=self.device)
            dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss_tensor.item() / self.world_size
        
        if self.rank == 0:
            print(f"评估完成，平均损失: {avg_loss:.4f}")
        
        return avg_loss
    
    def save(self, filename: str):
        """保存模型检查点（只在主进程中）"""
        if self.rank != 0:
            return
        
        filepath = os.path.join(self.save_dir, filename)
        
        # 获取原始模型（去掉DDP包装）
        model_to_save = self.model.module if self.is_distributed else self.model
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'training_history': self.training_history,
            'tokenizer_vocab': self.tokenizer.token_to_id,
            'model_config': {
                'model_type': 'Qwen3Transformer',
                'device': str(self.device),
                'batch_size': self.batch_size,
                'max_length': self.max_length,
                'vocab_size': self.tokenizer.vocab_size,
                'world_size': self.world_size,
                'is_distributed': self.is_distributed
            }
        }
        
        torch.save(checkpoint, filepath)
        print(f"模型已保存到: {filepath}")
    
    def load(self, filepath: str):
        """加载模型检查点"""
        if not os.path.exists(filepath):
            if self.rank == 0:
                print(f"检查点文件不存在: {filepath}")
            return False
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # 获取原始模型（去掉DDP包装）
        model_to_load = self.model.module if self.is_distributed else self.model
        
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.training_history = checkpoint['training_history']
        
        # 恢复分词器词汇表
        if 'tokenizer_vocab' in checkpoint:
            self.tokenizer.token_to_id = checkpoint['tokenizer_vocab']
            self.tokenizer.id_to_token = {v: k for k, v in self.tokenizer.token_to_id.items()}
        
        if self.rank == 0:
            print(f"模型已从 {filepath} 加载，当前轮数: {self.current_epoch}")
        
        return True
    
    def get_training_stats(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        return {
            'current_epoch': self.current_epoch,
            'best_loss': self.best_loss,
            'total_training_steps': len(self.training_history),
            'device': str(self.device),
            'training_history': self.training_history,
            'vocab_size': self.tokenizer.vocab_size,
            'max_length': self.max_length,
            'rank': self.rank,
            'world_size': self.world_size,
            'is_distributed': self.is_distributed
        }


def setup_distributed_training(local_rank: int, world_size: int):
    """
    设置分布式训练环境
    
    Args:
        local_rank: 本地GPU排名
        world_size: 总GPU数量
    """
    # 设置环境变量
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 初始化进程组
    dist.init_process_group(
        backend='nccl',  # 使用NCCL后端（GPU）
        init_method='env://',
        world_size=world_size,
        rank=local_rank
    )
    
    # 设置当前设备
    torch.cuda.set_device(local_rank)
    
    print(f"分布式训练环境初始化完成: rank {local_rank}/{world_size}")


def cleanup_distributed_training():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()
        print("分布式训练环境已清理")
