"""
Qwen3模型专用训练器
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, Any, List, Union
import os
from datetime import datetime
from collections import Counter

import sys
import os
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


class TextDataset(Dataset):
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


class Qwen3Trainer:
    """
    Qwen3模型专用训练器
    """
    
    def __init__(self, 
                 model: Qwen3Transformer,
                 dataset: TextDataset,
                 tokenizer: SimpleTokenizer,
                 optimizer: torch.optim.Optimizer,
                 criterion: nn.Module,
                 batch_size: int = 4,
                 device: str = "auto",
                 save_dir: str = "./qwen3_checkpoints",
                 max_length: int = 512):
        """
        初始化Qwen3训练器
        
        Args:
            model: Qwen3Transformer模型
            dataset: 文本数据集
            tokenizer: 分词器
            optimizer: 优化器
            criterion: 损失函数
            batch_size: 批次大小
            device: 设备 ("auto", "cpu", "cuda")
            save_dir: 保存目录
            max_length: 最大序列长度
        """
        self.model = model
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_size = batch_size
        self.max_length = max_length
        self.save_dir = save_dir
        
        # 自动选择设备
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # 将模型移到设备上
        self.model.to(self.device)
        
        # 创建数据加载器
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
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Qwen3训练器初始化完成，使用设备: {self.device}")
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"词汇表大小: {tokenizer.vocab_size}")
        print(f"最大序列长度: {max_length}")
    
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
        print(f"开始训练Qwen3模型，共 {num_epochs} 轮")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            epoch_loss = 0.0
            num_batches = 0
            
            # 设置为训练模式
            self.model.train()
            
            for batch_idx, (input_ids, target_ids) in enumerate(self.dataloader):
                # 将数据移到设备上
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                
                # 前向传播
                self.optimizer.zero_grad()
                outputs = self.model(input_ids)
                
                # 计算损失 - 语言模型损失
                # 将输出和目标都展平，用于计算交叉熵损失
                outputs_flat = outputs.view(-1, outputs.size(-1))
                targets_flat = target_ids.view(-1)
                
                # 忽略填充token的损失
                loss = self.criterion(outputs_flat, targets_flat)
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # 打印进度
                if batch_idx % log_every == 0:
                    print(f"Epoch {self.current_epoch}/{num_epochs}, "
                          f"Batch {batch_idx}/{len(self.dataloader)}, "
                          f"Loss: {loss.item():.4f}")
            
            # 计算平均损失
            avg_loss = epoch_loss / num_batches
            self.training_history.append({
                'epoch': self.current_epoch,
                'loss': avg_loss,
                'timestamp': datetime.now().isoformat()
            })
            
            print(f"Epoch {self.current_epoch} 完成，平均损失: {avg_loss:.4f}")
            
            # 保存检查点
            if self.current_epoch % save_every == 0:
                self.save(f"qwen3_checkpoint_epoch_{self.current_epoch}.pt")
            
            # 更新最佳损失
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.save("qwen3_best_model.pt")
        
        print("训练完成！")
    
    def eval(self, eval_dataset: Optional[TextDataset] = None):
        """
        评估模型
        
        Args:
            eval_dataset: 评估数据集，如果为None则使用训练数据集
        """
        if eval_dataset is None:
            eval_dataset = self.dataset
        
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
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                
                # 前向传播
                outputs = self.model(input_ids)
                
                # 计算损失
                outputs_flat = outputs.view(-1, outputs.size(-1))
                targets_flat = target_ids.view(-1)
                loss = self.criterion(outputs_flat, targets_flat)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"评估完成，平均损失: {avg_loss:.4f}")
        
        return avg_loss
    
    def generate_text(self, prompt: str, max_length: int = 100, temperature: float = 0.8) -> str:
        """
        生成文本
        
        Args:
            prompt: 输入提示
            max_length: 最大生成长度
            temperature: 温度参数，控制生成的随机性
        
        Returns:
            生成的文本
        """
        self.model.eval()
        
        # 编码输入
        input_ids = self.tokenizer.encode(prompt, max_length=self.max_length)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        
        generated_ids = input_ids.copy()
        
        with torch.no_grad():
            for _ in range(max_length):
                # 获取模型输出
                outputs = self.model(input_tensor)
                next_token_logits = outputs[0, -1, :] / temperature
                
                # 采样下一个token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                
                # 如果生成了结束标记，停止生成
                if next_token == self.tokenizer.eos_token_id:
                    break
                
                # 添加到生成序列
                generated_ids.append(next_token)
                input_tensor = torch.tensor([generated_ids], dtype=torch.long).to(self.device)
        
        # 解码生成的文本
        generated_text = self.tokenizer.decode(generated_ids)
        return generated_text
    
    def save(self, filename: str):
        """保存模型检查点"""
        filepath = os.path.join(self.save_dir, filename)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'training_history': self.training_history,
            'tokenizer_vocab': self.tokenizer.token_to_id,
            'model_config': {
                'model_type': 'Qwen3Transformer',
                'device': str(self.device),
                'batch_size': self.batch_size,
                'max_length': self.max_length,
                'vocab_size': self.tokenizer.vocab_size
            }
        }
        
        torch.save(checkpoint, filepath)
        print(f"模型已保存到: {filepath}")
    
    def load(self, filepath: str):
        """加载模型检查点"""
        if not os.path.exists(filepath):
            print(f"检查点文件不存在: {filepath}")
            return False
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.training_history = checkpoint['training_history']
        
        # 恢复分词器词汇表
        if 'tokenizer_vocab' in checkpoint:
            self.tokenizer.token_to_id = checkpoint['tokenizer_vocab']
            self.tokenizer.id_to_token = {v: k for k, v in self.tokenizer.token_to_id.items()}
        
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
            'max_length': self.max_length
        }
