import math
from turtle import hideturtle 

import torch 
import torch.nn.functional as F
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
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config 
        self.hidden_size = hidden_size if hidden_size is not None else config.hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states):
        # SwiGLU: SiLU(gate) * up
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        swiglu_output = self.act_fn(gate) * up
        output = self.down_proj(swiglu_output)
        return output 
    

class TopkRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_epr_tok 
        self.n_routed_experts = config.num_experts_epr_tok 
        self.routed_scaling_factor = config.routed_scaling_factor 
        self.n_group = config.n_group
        self.top_k_group = config.top_k_group
        self.norm_topk_prob = config.norm_topk_prob 

        self.weight = nn.Parameter(torch.empty(config.n_routed_experts, config.hidden_size))
        self.register_buffer("e_score_correction_bias", torch.zeros(config.n_routed_experts))

    @torch.no_grad()
    def get_top_k_indices(self, scores):
        scores_for_choice = scores.view(-1, self.n_routed_experts) + self.e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.top_k_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1,group_idx,1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
        top_k_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        return top_k_indices
    
    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        router_logits = F.linear(hidden_states.to(torch.float32), self.weight.type(torch.float32))
        scores = router_logits.sigmoid()
        top_k_indices = self.get_top_k_indices(scores)
        top_k_weights = scores.gather(1, top_k_indices)
        if self.norm_topk_prob:
            denominator = top_k_weights.sum(dim=-1, keepdim=True) + 1e-20
            top_k_weights /= denominator 
        top_k_weights = top_k_weights * self.routed_scaling_factor 
        return top_k_indices, top_k_weights