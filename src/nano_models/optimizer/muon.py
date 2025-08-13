"""
Muon 优化器实现

Muon 是一个结合了 Adam 和 SGD 优点的优化器，通过动态调整学习率和动量来提供更好的收敛性能。
"""

import torch
from torch.optim import Optimizer
from typing import Optional, Dict, Any, Tuple
import math


class Muon(Optimizer):
    """
    Muon 优化器实现
    
    参数:
        params: 需要优化的参数
        lr: 学习率 (默认: 1e-3)
        betas: 用于计算梯度和梯度平方的运行平均值的系数 (默认: (0.9, 0.999))
        eps: 数值稳定性常数 (默认: 1e-8)
        weight_decay: 权重衰减 (L2惩罚) (默认: 0)
        momentum: 动量系数 (默认: 0.9)
        nesterov: 是否使用 Nesterov 动量 (默认: False)
        adaptive_momentum: 是否使用自适应动量 (默认: True)
        warmup_steps: 预热步数 (默认: 0)
        decay_steps: 衰减步数 (默认: 0)
        min_lr: 最小学习率 (默认: 1e-6)
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        momentum: float = 0.9,
        nesterov: bool = False,
        adaptive_momentum: bool = True,
        warmup_steps: int = 0,
        decay_steps: int = 0,
        min_lr: float = 1e-6,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"无效的学习率: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"无效的 eps 值: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"无效的 beta 参数: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"无效的 beta 参数: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"无效的权重衰减值: {weight_decay}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"无效的动量值: {momentum}")
        if not 0.0 <= min_lr:
            raise ValueError(f"无效的最小学习率: {min_lr}")
        
        defaults = dict(
            lr=lr,
            base_lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            adaptive_momentum=adaptive_momentum,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            min_lr=min_lr,
        )
        super().__init__(params, defaults)
        
        # 全局状态
        self.step_count = 0
        self.base_lr = lr
        
    def __getstate__(self):
        return super().__getstate__()
    
    def __setstate__(self, state):
        super().__setstate__(state)
    
    def step(self, closure=None) -> float:  # type: ignore
        """
        执行单步优化
        
        参数:
            closure: 重新计算损失的闭包函数
            
        返回:
            损失值 (如果提供了 closure)
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Muon 不支持稀疏梯度')
                
                state = self.state[p]
                
                # 状态初始化
                if len(state) == 0:
                    state['step'] = 0
                    # 指数移动平均的梯度
                    state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    # 指数移动平均的梯度平方
                    state['exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    # 动量缓冲区
                    state['momentum_buffer'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # 权重衰减
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                
                # 更新指数移动平均
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # 计算偏差修正
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # 计算自适应学习率
                current_lr = self._get_adaptive_lr(group)
                
                # 计算步长
                step_size = current_lr / bias_correction1
                
                # 计算自适应动量
                if group['adaptive_momentum']:
                    adaptive_momentum = self._get_adaptive_momentum(group, exp_avg, exp_avg_sq)
                else:
                    adaptive_momentum = group['momentum']
                
                # 更新参数
                if group['nesterov']:
                    # Nesterov 动量
                    momentum_buffer = state['momentum_buffer']
                    momentum_buffer.mul_(adaptive_momentum).add_(exp_avg, alpha=step_size)
                    p.data.add_(momentum_buffer, alpha=adaptive_momentum)
                    p.data.add_(exp_avg, alpha=-step_size)
                else:
                    # 标准动量
                    momentum_buffer = state['momentum_buffer']
                    momentum_buffer.mul_(adaptive_momentum).add_(exp_avg, alpha=step_size)
                    p.data.add_(momentum_buffer, alpha=-1)
        
        self.step_count += 1
        return loss if loss is not None else 0.0
    
    def _get_adaptive_lr(self, group) -> float:
        """
        计算自适应学习率
        
        参数:
            group: 参数组
            
        返回:
            当前步的学习率
        """
        current_lr = group['lr']
        
        # 预热阶段
        if self.step_count < group['warmup_steps']:
            warmup_factor = self.step_count / group['warmup_steps']
            current_lr = group['base_lr'] * warmup_factor
        
        # 衰减阶段
        elif group['decay_steps'] > 0:
            decay_factor = 0.5 * (1 + math.cos(math.pi * (self.step_count - group['warmup_steps']) / group['decay_steps']))
            current_lr = group['base_lr'] * decay_factor
        
        # 确保学习率不低于最小值
        current_lr = max(current_lr, group['min_lr'])
        
        return current_lr
    
    def _get_adaptive_momentum(self, group, exp_avg, exp_avg_sq) -> float:
        """
        计算自适应动量
        
        参数:
            group: 参数组
            exp_avg: 梯度的指数移动平均
            exp_avg_sq: 梯度平方的指数移动平均
            
        返回:
            自适应动量值
        """
        base_momentum = group['momentum']
        
        # 基于梯度统计信息调整动量
        if exp_avg_sq.numel() > 0:
            # 计算梯度的相对变化
            grad_norm = torch.norm(exp_avg)
            grad_var = torch.var(exp_avg_sq)
            
            if grad_var > 0:
                # 梯度变化大时增加动量，变化小时减少动量
                stability_factor = torch.clamp(grad_norm / (torch.sqrt(grad_var) + group['eps']), 0.1, 10.0)
                adaptive_momentum = base_momentum * stability_factor.item()
                return min(adaptive_momentum, 0.99)  # 限制最大动量值
        
        return base_momentum
    
    def get_lr(self) -> float:
        """
        获取当前学习率
        
        返回:
            当前学习率
        """
        if len(self.param_groups) == 0:
            return 0.0
        
        # 返回第一个参数组的学习率作为参考
        return self.param_groups[0]['lr']
    
    def set_lr(self, lr: float):
        """
        设置学习率
        
        参数:
            lr: 新的学习率
        """
        if not 0.0 <= lr:
            raise ValueError(f"无效的学习率: {lr}")
        
        for group in self.param_groups:
            group['lr'] = lr
            group['base_lr'] = lr
    
    def get_momentum(self) -> float:
        """
        获取当前动量值
        
        返回:
            当前动量值
        """
        if len(self.param_groups) == 0:
            return 0.0
        
        return self.param_groups[0]['momentum']
    
    def set_momentum(self, momentum: float):
        """
        设置动量值
        
        参数:
            momentum: 新的动量值
        """
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"无效的动量值: {momentum}")
        
        for group in self.param_groups:
            group['momentum'] = momentum
    
    def reset_state(self):
        """
        重置优化器状态
        """
        self.step_count = 0
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if 'exp_avg' in state:
                    state['exp_avg'].zero_()
                if 'exp_avg_sq' in state:
                    state['exp_avg_sq'].zero_()
                if 'momentum_buffer' in state:
                    state['momentum_buffer'].zero_()
                if 'step' in state:
                    state['step'] = 0
    
    def extra_repr(self) -> str:
        """
        返回优化器的额外信息字符串
        """
        return f"lr={self.get_lr():.2e}, momentum={self.get_momentum():.2f}, step_count={self.step_count}"
