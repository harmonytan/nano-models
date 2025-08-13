from typing import Optional 

from numpy import sort
from sympy.ntheory import dra
import torch 
import torch.nn.functional as F
from torch.nn.modules import linear
import torch.utils.checkpoint 
from torch import nn 


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps 
    
    def forward(self, hidden_states):
        input_type = hidden_states.dtype 
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_type)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

class MLP(nn.Module):
    def __init__(self, hidden_size:int, intermediate_size:int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size)
        self.w2 = nn.Linear(intermediate_size, hidden_size)
        self.w3 = nn.Linear(hidden_size, intermediate_size)
    
    def forward(self, hidden_states):
        return self.w2(F.silu(self.w1(hidden_states)) * self.w3(hidden_states))

class Gate(nn.Module):
    def __init__(self, config):
        self.hidden_size = config.hidden_size
        self.topk = config.n_activated_experts 
        self.n_groups = config.n_experts_groups
        self.topk_groups = config.n_limited_groups 
        self.score_func = config.score_func
        self.route_scale = config.route_scale
        self.weight = nn.Parameter(torch.empty(config.n_routed_experts, config.hidden_size))
        self.bias = nn.Parameter(torch.empty(config.n_routed_experts)) if self.hidden_size == 7168 else None 

    def forward(self, hidden_states):
        scores = F.linear(hidden_states, self.weight)
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()
        original_scores = scores 
        if self.bias is not None:
            scores = scores + self.bias 
        if self.n_groups > 1:
            scores = scores.view(hidden_states.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = torch.ones(hidden_states.size(0), self.n_groups, dtype=torch.bool, device=scores.device).scatter_(1, indices, False)
            scores = scores.masked_fill(mask.unsqueeze(-1), float("inf")).flatten(1)
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights.type_as(hidden_states), indices 
    
class Expert(nn.Module):
    def __init__(self, hidden_size:int, intermediate_size:int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size)
        self.w2 = nn.Linear(intermediate_size, hidden_size)
        self.w3 = nn.Linear(hidden_size, intermediate_size)
    
    def forward(self, hidden_states):
        return self.w2(F.silu(self.w1(hidden_states)) * self.w3(hidden_states))

class MOE(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_routed_experts = config.n_routed_experts
        self.n_activated_experts = config.n_activated_experts
        self.gate = Gate(config)
        self.experts = nn.ModuleList(
            [Expert(config.hidden_size, config.moe_inter_dim) for i in range(self.n_routed_experts)]
        )
        self.share_experts = MLP(config.hidden_size, config.n_shared_experts * config.moe_inter_dim)

    def forward(self, hidden_states):
        shape = hidden_states.size()
        hidden_states = hidden_states.view(-1, self.hidden_size)
        weights, indices = self.gate(hidden_states)
        y = torch.zeros_like(hidden_states)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(len(self.experts)):
            pass

        
def apply_rotary_embedding(hidden_states, position_ids=None, head_dim=None, max_position_embeddings=4096):
    if head_dim is None:
        head_dim = hidden_states.size(-1)
    
    # 确保head_dim是偶数
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim必须是偶数，当前为{head_dim}")
    
    # 生成位置编码
    if position_ids is None:
        seq_len = hidden_states.size(1)
        position_ids = torch.arange(seq_len, device=hidden_states.device)
    
    # 计算旋转角度
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=hidden_states.device).float() / head_dim))
    
    # 计算sin和cos值
    sinusoid_inp = torch.einsum("i,j->ij", position_ids.float(), inv_freq)
    sin = torch.sin(sinusoid_inp)
    cos = torch.cos(sinusoid_inp)
    
    # 重塑hidden_states以便应用旋转
    batch_size, seq_len, num_heads, head_dim = hidden_states.shape
    hidden_states = hidden_states.view(batch_size, seq_len, num_heads, head_dim // 2, 2)
    
    # 分离实部和虚部
    x1, x2 = hidden_states[..., 0], hidden_states[..., 1]
    
    # 应用旋转
    rotated_x1 = x1 * cos.unsqueeze(1).unsqueeze(0) - x2 * sin.unsqueeze(1).unsqueeze(0)
    rotated_x2 = x1 * sin.unsqueeze(1).unsqueeze(0) + x2 * cos.unsqueeze(1).unsqueeze(0)
    
    # 重新组合
    rotated_hidden_states = torch.stack([rotated_x1, rotated_x2], dim=-1)
    rotated_hidden_states = rotated_hidden_states.view(batch_size, seq_len, num_heads, head_dim)
    
    return rotated_hidden_states


def get_rotary_embeddings(seq_len, head_dim, device=None, dtype=None):
    if device is None:
        device = torch.device('cpu')
    if dtype is None:
        dtype = torch.float32
    
    # 确保head_dim是偶数
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim必须是偶数，当前为{head_dim}")
    
    # 计算旋转角度
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    
    # 计算sin和cos值
    position_ids = torch.arange(seq_len, device=device, dtype=dtype)
    sinusoid_inp = torch.einsum("i,j->ij", position_ids, inv_freq)
    
    sin = torch.sin(sinusoid_inp)
    cos = torch.cos(sinusoid_inp)
    
    return sin, cos


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    # 获取序列长度
    seq_len = q.size(1)
    
    # 如果提供了position_ids，使用它们来索引cos和sin
    if position_ids is not None:
        cos = cos[position_ids]
        sin = sin[position_ids]
    else:
        cos = cos[:seq_len]
        sin = sin[:seq_len]
    
    # 扩展维度以匹配q和k的形状
    cos = cos.unsqueeze(0).unsqueeze(1)  # (1, seq_len, 1, head_dim//2)
    sin = sin.unsqueeze(0).unsqueeze(1)  # (1, seq_len, 1, head_dim//2)
    
    # 应用旋转到q
    q_rotated = apply_rotary_to_tensor(q, cos, sin)
    
    # 应用旋转到k
    k_rotated = apply_rotary_to_tensor(k, cos, sin)
    
    return q_rotated, k_rotated


def apply_rotary_to_tensor(tensor, cos, sin):
    # 获取张量的形状
    batch_size, seq_len, num_heads, head_dim = tensor.shape
    
    # 确保head_dim是偶数
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim必须是偶数，当前为{head_dim}")
    
    # 重塑张量以便应用旋转
    tensor = tensor.view(batch_size, seq_len, num_heads, head_dim // 2, 2)
    
    # 分离实部和虚部
    x1, x2 = tensor[..., 0], tensor[..., 1]
    
    # 应用旋转
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos
    
    # 重新组合
    rotated_tensor = torch.stack([rotated_x1, rotated_x2], dim=-1)
    rotated_tensor = rotated_tensor.view(batch_size, seq_len, num_heads, head_dim)
    
    return rotated_tensor 

class MLA(nn.Module):
    def __init__(self,config) :
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_heads = config.n_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim

        self.wq_a = nn.Linear(self.hidden_size, self.q_lora_rank)
        self.q_norm = RMSNorm(config.hidden_size)
        self.wq_b = nn.Linear(self.q_lora_rank, self.hidden_size)

        self.wkv_a = nn.Linear(self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.hidden_size)
        self.wkv_b = nn.Linear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.hidden_size)
        self.softmax_scale = self.qk_head_dim ** -0.5

    def forward(self, hidden_states: torch.Tensor, mask:Optional[torch.Tensor]):
        batch_size, seq_length, _ = hidden_states.size()
        
        query = self.wq_b(self.q_norm(self.wq_a(hidden_states)))
        query = query.view(batch_size, seq_length, self.n_heads, self.qk_head_dim)
        query_nope, query_rope = torch.split(query, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        query_rope = apply_rotary_embedding(query_rope)

        key_value = self.wkv_a(hidden_states)
        key_value, key_rope = torch.split(key_value, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        key_rope = apply_rotary_embedding(key_rope)

        query = torch.cat([query_nope, query_rope], dim=-1)
        key_value = self.wkv_b(self.kv_norm(key_value)).view(batch_size, seq_length, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
        key_nope, value = torch.split(key_value, [self.qk_nope_head_dim, self.v_head_dim], dim =-1)
        key = torch.cat([key_nope, key_rope.expand(-1,-1,self.n_heads, -1)], dim =-1)
        scores = torch.einsum("bshd,bthd->bsht", query, key) * self.softmax_scale

        if mask is not None:
            scores += mask.unsqueeze(1)
        scores = scores.softmax(dim=-1, dtype=torch.float32)
        hidden_states = torch.einsum("bsht, bthd -> bshd",scores, value)
        hidden_states = self.wo(hidden_states)
        return hidden_states
    
class Block(nn.Module):
    def __init__(self, config, layer_id) -> None:
        super().__init__()
        self.attn = MLA(config)
        self.ffn = MOE(config)
        self.attn_norm = RMSNorm(config.hidden_size)
        self.ffn_norm = RMSNorm(config.hidden_size)
    
    def forward(self, hidden_states, mask):
        hidden_states = hidden_states + self.attn(self.attn_norm(hidden_states), mask)
        hidden_states = hidden_states + self.ffn(self.ffn_norm(hidden_states))
        return hidden_states

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config.vacab_size, config.hidden_size)
        self.layers = nn.ModuleList()
        for layer_id in range(config.decoder_layers):
            self.layers.append(Block(config, layer_id))
        self.norm = RMSNorm(config.hidden_size)
        self.logit_head = nn.Linear(config.hidden_size, config.vocab_size)
    
    def forward(self, tokens, mask ):

        hidden_states = self.embed(tokens)
        for layer in self.layers:
            hidden_states = layer(hidden_states, mask)
        hidden_states = self.norm(hidden_states)
        logits = self.logit_head(hidden_states)
        return logits 





