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
# Global Feature Cross-Attention
# =========================
class GlobalFeatureAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8, num_queries: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_queries = num_queries
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, d_model) * 0.02)
        self.cross_attn = CrossModalAttention(d_model, num_heads, dropout)
        self.output_proj = nn.Sequential(
            nn.Linear(d_model * num_queries, d_model),
            nn.LayerNorm(d_model)
        )
        
    def forward(self, seq: torch.Tensor, global_feat: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = seq.shape
        
        global_kv = global_feat.unsqueeze(1)
        queries = self.query_tokens.expand(B, -1, -1)
        
        attended = self.cross_attn(queries, global_kv, global_kv)
        
        attended_flat = attended.reshape(B, -1)
        context = self.output_proj(attended_flat).unsqueeze(1)
        context = context.expand(-1, L, -1)
        
        if mask is not None:
            context = context * mask.unsqueeze(-1)
        
        return context

# =========================
# 消融实验：ESM2 + 单全局特征融合模型
# =========================
class AblationESM2SingleGlobal(nn.Module):
    """
    消融实验模型：只使用 ESM2 序列特征 + 一个全局特征
    可选择使用 Foldseek 或 ProstT5 作为全局特征
    """
    def __init__(
        self,
        esm2_dim: int = 1280,
        global_dim: int = 1024,  # Foldseek 或 ProstT5 的维度
        d_model: int = 512,
        num_heads: int = 8,
        num_experts: int = 8,
        top_k_experts: int = 2,
        num_fusion_layers: int = 4,
        dropout: float = 0.1,
        global_feature_name: str = "foldseek"  # "foldseek" 或 "prost5"
    ):
        super().__init__()
        self.global_feature_name = global_feature_name
        
        # ESM2 序列编码器
        self.esm2_encoder = ModalityEncoder(esm2_dim, d_model, dropout)
        
        # 全局特征编码器
        self.global_encoder = ModalityEncoder(global_dim, d_model, dropout)
        
        # 位置编码（仅用于序列特征）
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        
        # 全局特征注意力
        self.global_attn = GlobalFeatureAttention(
            d_model, num_heads, num_queries=4, dropout=dropout
        )
        
        # 双模态融合（ESM2 + 全局特征）
        self.initial_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )
        
        # 多层 Transformer + MoE
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "self_attn": CrossModalAttention(d_model, num_heads, dropout),
                "moe": MixtureOfExperts(d_model, num_experts, top_k_experts, dropout=dropout)
            }) for _ in range(num_fusion_layers)
        ])
        
        self.final_ln = nn.LayerNorm(d_model)

    def forward(self, esm2_feat, global_feat, temp=1.0):
        """
        Args:
            esm2_feat: [B, L, esm2_dim] - ESM2 序列特征
            global_feat: [B, global_dim] - 全局特征 (Foldseek 或 ProstT5)
            temp: MoE 温度参数
        Returns:
            fused: [B, L, d_model] - 融合后的特征
            total_lb: load balance loss
        """
        B, L, _ = esm2_feat.shape
        
        # 序列 mask
        mask = (esm2_feat.abs().sum(dim=-1) > 0)  # [B, L]

        # 编码 ESM2 序列特征
        seq = self.pos_enc(self.esm2_encoder(esm2_feat))  # [B, L, d_model]
        
        # 编码全局特征
        global_encoded = self.global_encoder(global_feat)  # [B, d_model]

        # 使用注意力机制融合全局特征到序列
        global_ctx = self.global_attn(seq, global_encoded, mask)  # [B, L, d_model]

        # 初始融合：ESM2 + 全局上下文
        fused = self.initial_fusion(
            torch.cat([seq, global_ctx], dim=-1)
        )  # [B, L, d_model]
        
        fused = fused * mask.unsqueeze(-1)

        # 多层 Transformer + MoE
        total_lb = 0.0
        for layer in self.layers:
            fused = layer["self_attn"](fused, fused, fused, mask)
            fused = fused * mask.unsqueeze(-1)
            fused, lb = layer['moe'](fused, temp=temp)
            total_lb += lb

        fused = self.final_ln(fused)
        return fused, total_lb


# =========================
# 使用示例
# =========================
if __name__ == "__main__":
    # 创建模型实例
    
    # 选项1：ESM2 + Foldseek
    model_foldseek = AblationESM2SingleGlobal(
        esm2_dim=1280,
        global_dim=1024,  # Foldseek 维度
        d_model=512,
        num_heads=8,
        num_experts=4,
        top_k_experts=2,
        num_fusion_layers=2,
        dropout=0.2,
        global_feature_name="foldseek"
    )
    
    # 选项2：ESM2 + ProstT5
    model_prost = AblationESM2SingleGlobal(
        esm2_dim=1280,
        global_dim=512,  # ProstT5 维度
        d_model=512,
        num_heads=8,
        num_experts=4,
        top_k_experts=2,
        num_fusion_layers=2,
        dropout=0.2,
        global_feature_name="prost5"
    )
    
    # 测试
    B, L = 4, 100
    esm2 = torch.randn(B, L, 1280)
    foldseek = torch.randn(B, 1024)
    prost5 = torch.randn(B, 512)
    
    # 使用 Foldseek
    out1, lb1 = model_foldseek(esm2, foldseek)
    print(f"ESM2+Foldseek output shape: {out1.shape}, LB loss: {lb1.item():.4f}")
    
    # 使用 ProstT5
    out2, lb2 = model_prost(esm2, prost5)
    print(f"ESM2+ProstT5 output shape: {out2.shape}, LB loss: {lb2.item():.4f}")