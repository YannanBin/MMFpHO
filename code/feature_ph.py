import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

# =========================
# Positional Encoding
# =========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
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
# Cross Attention (Pre-LN)
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
    ):
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

    def forward(self, x: torch.Tensor):
        return self.net(x)

# =========================
# Mixture of Experts (Sample-level routing)
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
        weights = F.softmax(scores, dim=1)

        x_sample = torch.sum(x_norm * weights.unsqueeze(-1), dim=1)
        gate_logits = self.gate(x_sample)

        if self.training:
            progress = min(1.0, self.step.item() / self.warmup_steps)
            temp = temp * (1.0 - progress) + self.min_temp * progress
            self.step += 1

        probs = F.softmax(gate_logits / temp, dim=-1)

        out = torch.zeros_like(x_norm)
        for i, expert in enumerate(self.experts):
            out += probs[:, i:i+1].unsqueeze(1) * expert(x_norm)

        lb_loss = self.num_experts * (probs.mean(0) ** 2).sum()

        out = out * token_mask.unsqueeze(-1)
        return out + residual, lb_loss

# =========================
# Global Feature Attention
# =========================
class GlobalFeatureAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8, num_queries: int = 4, dropout: float = 0.1):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, d_model) * 0.02)
        self.cross_attn = CrossModalAttention(d_model, num_heads, dropout)
        self.output_proj = nn.Sequential(
            nn.Linear(d_model * num_queries, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, seq, global_feat, mask=None):
        B, L, D = seq.shape
        queries = self.query_tokens.expand(B, -1, -1)
        kv = global_feat.unsqueeze(1)

        attended = self.cross_attn(queries, kv, kv)
        context = self.output_proj(attended.reshape(B, -1)).unsqueeze(1)
        context = context.expand(-1, L, -1)

        if mask is not None:
            context = context * mask.unsqueeze(-1)
        return context

# =========================
# Cross-modal Fusion Model (Sample-aware modality gate)
# =========================
class CrossModalFusionWithMoE(nn.Module):
    def __init__(
        self,
        esm2_dim=1280,
        foldseek_dim=1024,
        prost5_dim=1024,
        d_model=512,
        num_heads=8,
        num_experts=8,
        top_k_experts=2,
        num_fusion_layers=4,
        dropout=0.1,
    ):
        super().__init__()

        self.esm2_encoder = ModalityEncoder(esm2_dim, d_model, dropout)
        self.foldseek_encoder = ModalityEncoder(foldseek_dim, d_model, dropout)
        self.prost5_encoder = ModalityEncoder(prost5_dim, d_model, dropout)

        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        self.struct_attn = GlobalFeatureAttention(d_model, num_heads, dropout=dropout)
        self.phys_attn = GlobalFeatureAttention(d_model, num_heads, dropout=dropout)

        # ⭐ Sample-aware modality gate
        self.modality_gate = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3),
        )

        self.initial_fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "self_attn": CrossModalAttention(d_model, num_heads, dropout),
                "moe": MixtureOfExperts(d_model, num_experts, top_k_experts, dropout=dropout)
            })
            for _ in range(num_fusion_layers)
        ])

        self.final_ln = nn.LayerNorm(d_model)

    def forward(self, esm2_feat, foldseek_feat, prost5_feat, temp=1.0):
        B, L, _ = esm2_feat.shape
        mask = (esm2_feat.abs().sum(dim=-1) > 0)

        seq = self.pos_enc(self.esm2_encoder(esm2_feat))
        struct_global = self.foldseek_encoder(foldseek_feat)
        phys_global = self.prost5_encoder(prost5_feat)

        struct_ctx = self.struct_attn(seq, struct_global, mask)
        phys_ctx = self.phys_attn(seq, phys_global, mask)

        # sample-level summaries
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
        seq_sample = (seq * mask.unsqueeze(-1)).sum(dim=1) / denom

        sample_concat = torch.cat(
            [seq_sample, struct_global, phys_global], dim=-1
        )

        modality_weights = F.softmax(self.modality_gate(sample_concat), dim=-1)

        fused_tokens = torch.cat(
            [
                seq * modality_weights[:, 0:1].unsqueeze(1),
                struct_ctx * modality_weights[:, 1:2].unsqueeze(1),
                phys_ctx * modality_weights[:, 2:3].unsqueeze(1),
            ],
            dim=-1
        )

        fused = self.initial_fusion(fused_tokens)
        fused = fused * mask.unsqueeze(-1)

        total_lb = 0.0
        for layer in self.layers:
            fused = layer["self_attn"](fused, fused, fused, mask)
            fused = fused * mask.unsqueeze(-1)
            fused, lb = layer["moe"](fused, temp)
            total_lb += lb

        fused = self.final_ln(fused)
        return fused, total_lb
