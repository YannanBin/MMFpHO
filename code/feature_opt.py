import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

# =========================
# Positional Encoding
# =========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# =========================
# Modality Encoder
# =========================
class ModalityEncoder(nn.Module):
    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        x = self.ln(x)
        return self.dropout(x)

# =========================
# Cross-Attention (Pre-LN)
# =========================
class CrossModalAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.ln = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = query.size(0)
        residual = query
        
        query = self.ln(query)
        
        Q = self.q_proj(query).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ V
        out = out.transpose(1, 2).reshape(B, -1, self.d_model)
        out = self.out_proj(out)
        out = self.dropout(out)
        
        return out + residual

# =========================
# Expert
# =========================
class Expert(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# =========================
# Mixture of Experts
# =========================
class MixtureOfExperts(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_experts: int = 8,
        top_k: int = 2,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model

        self.num_experts = num_experts
        self.top_k = top_k

        self.experts = nn.ModuleList(
            [Expert(d_model, d_ff, dropout) for _ in range(num_experts)]
        )

        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_experts),
        )

        self.ln = nn.LayerNorm(d_model)
        self.pooler = nn.Linear(d_model, 1, bias=False)

        self.register_buffer("step", torch.zeros(1, dtype=torch.long))
        self.warmup_steps = 2000
        self.min_temp = 0.5

    def forward(self, x: torch.Tensor, temp: float = 1.0):
        B, L, D = x.shape
        residual = x
        token_mask = (x.abs().sum(dim=-1) > 0)
        x_norm = self.ln(x)

        scores = self.pooler(x_norm).squeeze(-1)
        scores = scores.masked_fill(token_mask == 0, -1e9)
        pooling_weights = F.softmax(scores, dim=1)
        x_sample = torch.sum(x_norm * pooling_weights.unsqueeze(-1), dim=1)

        gate_logits = self.gate(x_sample)

        if self.training:
            progress = min(1.0, self.step.item() / self.warmup_steps)
            annealed_temp = temp * (1.0 - progress) + self.min_temp * progress
            self.step += 1
        else:
            annealed_temp = self.min_temp

        use_soft = self.training and (self.step <= self.warmup_steps)

        if use_soft:
            probs = F.softmax(gate_logits / annealed_temp, dim=-1)
            out = torch.zeros_like(x_norm)
            for eid in range(self.num_experts):
                expert_out = self.experts[eid](x_norm)
                out += probs[:, eid:eid + 1].unsqueeze(1) * expert_out
            f = probs.mean(0)
            P = probs.mean(0)
            lb_loss = self.num_experts * (f * P).sum()
        else:
            topk_logits, topk_idx = torch.topk(gate_logits, self.top_k, dim=-1)
            topk_gate = F.softmax(topk_logits, dim=-1)
            out = torch.zeros_like(x_norm)
            expert_freqs = torch.zeros(self.num_experts, device=x.device)

            for k in range(self.top_k):
                expert_idx = topk_idx[:, k]
                gate_w = topk_gate[:, k].unsqueeze(1).unsqueeze(2)
                for eid in range(self.num_experts):
                    mask = (expert_idx == eid)
                    if mask.any():
                        expert_out = self.experts[eid](x_norm[mask])
                        out[mask] += gate_w[mask] * expert_out
                        expert_freqs[eid] += mask.sum()

            expert_freqs = expert_freqs / B
            gate_prob = F.softmax(gate_logits, dim=-1).mean(0)
            lb_loss = self.num_experts * (expert_freqs * gate_prob).sum()

        out = out * token_mask.unsqueeze(-1)
        out = out + residual
        return out, lb_loss


# =========================
# ✨ 全局特征增强模块（轻量级）
# =========================
class GlobalFeatureEnhancer(nn.Module):
    """
    将全局特征作为辅助信号注入序列特征
    使用 FiLM (Feature-wise Linear Modulation) 机制
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        # 生成调制参数 (scale & shift)
        self.film_gen = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model * 2)
        )
        
    def forward(self, seq: torch.Tensor, global_feat: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        seq: [B, L, D] 序列特征
        global_feat: [B, D] 全局特征
        
        Returns: [B, L, D] 增强后的序列特征
        """
        B, L, D = seq.shape
        
        # 生成调制参数
        film_params = self.film_gen(global_feat)  # [B, 2D]
        gamma, beta = torch.chunk(film_params, 2, dim=-1)  # 各自 [B, D]
        
        # 广播到序列维度
        gamma = gamma.unsqueeze(1)  # [B, 1, D]
        beta = beta.unsqueeze(1)    # [B, 1, D]
        
        # FiLM: y = γ * x + β (element-wise)
        enhanced = gamma * seq + beta
        
        # 应用 mask
        if mask is not None:
            enhanced = enhanced * mask.unsqueeze(-1)
        
        return enhanced


# =========================
# ✨ 序列主导的融合模型
# =========================
class SequenceDominantFusion(nn.Module):
    """
    架构理念：
    1. ESM2 序列特征为主干 (backbone)
    2. Foldseek/ProstT5 通过轻量级 FiLM 增强序列特征
    3. MoE 层仅作用于增强后的序列特征
    """
    def __init__(
        self,
        esm2_dim: int = 1280,
        foldseek_dim: int = 1024,
        prost5_dim: int = 512,
        d_model: int = 512,
        num_heads: int = 8,
        num_experts: int = 6,
        top_k_experts: int = 2,
        num_fusion_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        # 1️⃣ 编码器（序列特征更深）
        self.esm2_encoder = nn.Sequential(
            nn.Linear(esm2_dim, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # 全局特征编码器（更轻量）
        self.foldseek_encoder = nn.Sequential(
            nn.Linear(foldseek_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.prost5_encoder = nn.Sequential(
            nn.Linear(prost5_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        
        # 2️⃣ 全局特征增强器（轻量级）
        self.struct_enhancer = GlobalFeatureEnhancer(d_model, dropout)
        self.phys_enhancer = GlobalFeatureEnhancer(d_model, dropout)
        
        # 3️⃣ 序列主干网络（Self-Attention + MoE）
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "self_attn": CrossModalAttention(d_model, num_heads, dropout),
                "moe": MixtureOfExperts(d_model, num_experts, top_k_experts, dropout=dropout)
            }) for _ in range(num_fusion_layers)
        ])
        
        self.final_ln = nn.LayerNorm(d_model)
        
        # 4️⃣ 可学习的增强权重（控制全局特征的影响）
        self.enhancement_weight = nn.Parameter(torch.tensor([0.3, 0.3]))  # 初始为较小值

    def forward(self, esm2_feat, foldseek_feat, prost5_feat, temp=1.0):
        """
        esm2_feat: [B, L, 1280]
        foldseek_feat: [B, 1024]
        prost5_feat: [B, 512]
        """
        B, L, _ = esm2_feat.shape
        
        # Padding mask
        mask = (esm2_feat.abs().sum(dim=-1) > 0)  # [B, L]
        
        # 1️⃣ 编码各模态特征
        seq = self.esm2_encoder(esm2_feat)  # [B, L, D]
        seq = self.pos_enc(seq)
        
        struct_global = self.foldseek_encoder(foldseek_feat)  # [B, D]
        phys_global = self.prost5_encoder(prost5_feat)        # [B, D]
        
        # 2️⃣ 全局特征增强序列（轻量级注入）
        struct_enhancement = self.struct_enhancer(seq, struct_global, mask)
        phys_enhancement = self.phys_enhancer(seq, phys_global, mask)
        
        # 使用可学习权重融合
        weights = torch.sigmoid(self.enhancement_weight)  # 限制在 [0,1]
        seq = seq + weights[0] * struct_enhancement + weights[1] * phys_enhancement
        
        # 应用 mask
        seq = seq * mask.unsqueeze(-1)
        
        # 3️⃣ 序列主干网络处理
        total_lb = 0.0
        for layer in self.layers:
            # Self-attention
            seq = layer["self_attn"](seq, seq, seq, mask)
            seq = seq * mask.unsqueeze(-1)
            
            # MoE
            seq, lb = layer['moe'](seq, temp=temp)
            total_lb += lb
        
        # 4️⃣ 最终归一化
        seq = self.final_ln(seq)
        
        return seq, total_lb