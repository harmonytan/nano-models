from dataclasses import dataclass
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

@dataclass
class Qwen3Config:
    """
    Qwen3 模型配置类
    
    参数说明:
    - hidden_size: 隐藏层维度，决定了模型的表示能力
    - vocab_size: 词汇表大小，决定了模型能处理的词汇数量
    - intermediate_size: MLP中间层维度，通常为hidden_size的4倍
    - num_decoder_layer: 解码器层数，决定了模型的深度和复杂度
    - head_dim: 注意力头维度，每个注意力头的特征维度
    - num_attention_heads: 注意力头数量，多头注意力的并行度
    - num_key_value_heads: KV注意力头数量，用于分组查询注意力(GQA)
    - num_key_value_groups: KV分组数量，GQA中的分组参数
    - attention_bias: 是否在注意力层中使用偏置项
    - rms_norm_eps: RMS归一化的epsilon值，防止除零错误
    - max_position_embeddings: 最大位置编码长度，默认2048
    """
    hidden_size: int
    vocab_size: int
    intermediate_size: int 
    num_decoder_layer: int
    head_dim: int
    num_attention_heads: int    
    num_key_value_groups: int
    num_key_value_heads: int
    attention_bias: bool
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 2048
    
    @classmethod
    def from_preset(cls, model_size: str):
        """
        从预设模型大小创建配置
        
        支持的模型大小:
        - 'tiny': 最小模型，用于测试和调试
        - 'small': 小模型，适合资源受限环境
        - 'medium': 中等模型，平衡性能和资源
        - 'large': 大模型，高性能但需要更多资源
        """
        presets = {
            'tiny': {
                'hidden_size': 256,
                'vocab_size': 32000,
                'intermediate_size': 1024,
                'num_decoder_layer': 6,
                'head_dim': 64,
                'num_attention_heads': 4,
                'num_key_value_heads': 4,
                'num_key_value_groups': 1,
                'attention_bias': False,
            },
            'small': {
                'hidden_size': 512,
                'vocab_size': 32000,
                'intermediate_size': 2048,
                'num_decoder_layer': 12,
                'head_dim': 64,
                'num_attention_heads': 8,
                'num_key_value_heads': 8,
                'num_key_value_groups': 1,
                'attention_bias': False,
            },
            'medium': {
                'hidden_size': 1024,
                'vocab_size': 32000,
                'intermediate_size': 4096,
                'num_decoder_layer': 24,
                'head_dim': 64,
                'num_attention_heads': 16,
                'num_key_value_heads': 16,
                'num_key_value_groups': 1,
                'attention_bias': False,
            },
            'large': {
                'hidden_size': 2048,
                'vocab_size': 32000,
                'intermediate_size': 8192,
                'num_decoder_layer': 32,
                'head_dim': 64,
                'num_attention_heads': 32,
                'num_key_value_heads': 32,
                'num_key_value_groups': 1,
                'attention_bias': False,
            }
        }
        
        if model_size not in presets:
            raise ValueError(f"不支持的模型大小: {model_size}. 支持的大小: {list(presets.keys())}")
        
        return cls(**presets[model_size])
    
    @property
    def total_params(self) -> int:
        """计算模型总参数数量（估算）"""
        # 嵌入层
        embed_params = self.vocab_size * self.hidden_size
        
        # 每个解码器层
        layer_params = (
            # 注意力层
            self.hidden_size * self.hidden_size * 4 +  # q, k, v, o projections
            self.hidden_size * self.hidden_size * 2 +  # q_norm, k_norm
            # MLP层
            self.hidden_size * self.intermediate_size * 3 +  # w1, w2, w3
            self.hidden_size * 2  # pre_layernorm, post_attention_layernorm
        )
        
        # 输出层
        output_params = self.hidden_size * self.vocab_size
        
        total = embed_params + (layer_params * self.num_decoder_layer) + output_params
        return total
    
    @property
    def model_size_mb(self) -> float:
        """估算模型大小（MB）"""
        return self.total_params * 4 / (1024 * 1024)  # 假设float32
    
    def __str__(self) -> str:
        return f"Qwen3Config(hidden_size={self.hidden_size}, layers={self.num_decoder_layer}, heads={self.num_attention_heads}, params={self.total_params:,})"


class Qwen3RMSNorm(nn.Module):
    def __init__(self, dim:int, eps:float=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps 
    
    def forward(self, x: torch.Tensor):
        input_type = x.dtype 
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        # 确保 weight 的维度与输入张量的最后一个维度匹配
        weight = self.weight.view(1, 1, -1) if x.dim() == 3 else self.weight.view(1, 1, 1, -1)
        return weight * x.to(input_type)

class Qwen3MLP(nn.Module):
    def __init__(self, config:Qwen3Config):
        super().__init__()
        in_dim = config.hidden_size
        inter_dim = config.intermediate_size
        self.activate_fn = F.silu
        self.w1 = nn.Linear(in_dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, in_dim, bias=False)
        self.w3 = nn.Linear(in_dim, inter_dim, bias=False)
    
    def forward(self, x):
        return  self.w2(self.activate_fn(self.w1(x)) * self.w3(x))

def rotary_half(x):
    x1 = x[...,:x.shape[-1]//2]
    x2 = x[...,x.shape[-1]//2:]
    return torch.cat([-x2,x1],dim=-1)

def apply_rotary_pos_emb(q, k, sin, cos, unsqueeze_dim=1):
    # q, k: [batch_size, num_heads, seq_length, head_dim]
    # sin, cos: [seq_length, head_dim]
    
    # 扩展维度以匹配 q, k 的形状
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_length, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_length, head_dim]
    
    q_embed = (q * cos) + (rotary_half(q) * sin)
    k_embed = (k * cos) + (rotary_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(x, n_rep:int):
    batch_size, num_key_value_groups, seq_length, head_dim = x.size()
    if n_rep == 1:
        return x 
    x = x[:,:,None,:,:].expand(batch_size,num_key_value_groups,n_rep,seq_length,head_dim)
    return x.reshape(batch_size, num_key_value_groups * n_rep, seq_length, head_dim)

class Qwen3Attention(nn.Module):
    def __init__(self, config:Qwen3Config, layer_idx:int, max_position_embeddings:int=2048):
        super().__init__()
        self.config = config 
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_attention_heads
        self.num_key_value_groups = config.num_key_value_groups 
        self.scaling = self.head_dim ** -0.5
        self.max_position_embeddings = max_position_embeddings

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)

        self.q_norm = Qwen3RMSNorm(config.hidden_size,config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(config.hidden_size, config.rms_norm_eps)
        
        # Register position embeddings as buffers
        self._init_rope()
    
    def _init_rope(self):
        """Initialize RoPE position embeddings"""
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        position = torch.arange(self.max_position_embeddings)
        freqs = torch.einsum("i,j->ij", position, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(
        self, 
        x:torch.Tensor
    ):
        batch_size, seq_length, _ = x.size()

        # Project and normalize, then reshape to multi-head format
        query_states = self.q_norm(self.q_proj(x)).view(batch_size, seq_length, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(x)).view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(x).view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Get cached position embeddings
        cos = self.cos_cached[:seq_length]  # type: ignore
        sin = self.sin_cached[:seq_length]  # type: ignore
        query_states,key_states = apply_rotary_pos_emb(query_states,key_states,sin,cos)

        # Repeat KV heads for grouped query attention
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        
        # Compute attention weights: [batch_size, num_heads, seq_length, seq_length]
        # query_states: [batch_size, num_attention_heads, seq_length, head_dim]
        # key_states: [batch_size, num_attention_heads, seq_length, head_dim] (after repeat_kv)
        # We need to compute attention between all positions: [batch_size, num_heads, seq_length, seq_length]
        attn_weights = torch.einsum('bhsd,bhtd->bhst', query_states, key_states) * self.scaling
        

        
        # Apply casual mask for causal attention
        # attn_weights shape: [batch_size, num_heads, seq_length, seq_length]
        casual_mask = torch.triu(torch.ones(seq_length, seq_length, device=x.device, dtype=torch.bool), diagonal=1)
        casual_mask = casual_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_length, seq_length]
        casual_mask = casual_mask.expand(batch_size, self.num_attention_heads, seq_length, seq_length)  # [batch_size, num_heads, seq_length, seq_length]
        attn_weights = attn_weights.masked_fill(casual_mask, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.einsum("bhss,bhsd -> bhsd", attn_weights, value_states)
        attn_output = attn_output.transpose(1,2).contiguous().reshape(batch_size,seq_length,-1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output

class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config:Qwen3Config, layer_idx:int) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3Attention(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.pre_layernorm = Qwen3RMSNorm(self.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(self.hidden_size, config.rms_norm_eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the decoder layer
        
        Args:
            x: Input tensor [batch_size, seq_length, hidden_size]
        
        Returns:
            Output tensor [batch_size, seq_length, hidden_size]
        """
        # Self-attention block with residual connection
        residual = x 
        x = self.pre_layernorm(x)
        x = self.self_attn(x)
        x = x + residual 

        # MLP block with residual connection
        residual = x 
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = x + residual 
        
        return x 

class Qwen3Transformer(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.decoder_list = nn.ModuleList([
            Qwen3DecoderLayer(config, layer_idx) 
            for layer_idx in range(config.num_decoder_layer)
        ])
        self.final_layernorm = Qwen3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def forward(self, input_ids: torch.LongTensor):
        """
        Forward pass of the transformer
        
        Args:
            input_ids: Input token IDs [batch_size, seq_length]
            attention_mask: Optional attention mask [batch_size, seq_length]
        
        Returns:
            logits: Output logits [batch_size, seq_length, vocab_size]
        """
        batch_size, seq_length = input_ids.size()
        
        # Token embedding
        x = self.embed(input_ids)
        
        # Apply decoder layers
        for decoder_layer in self.decoder_list:
            x = decoder_layer(x)
        
        # Final layer normalization
        x = self.final_layernorm(x)
        
        # Language model head
        logits = self.lm_head(x)
        
        return logits







        

        

        




