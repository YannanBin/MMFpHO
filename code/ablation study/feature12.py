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
# ✨ 增强版 FiLM 模块
# =========================
class EnhancedFiLM(nn.Module):
    """
    改进的 FiLM 机制：
    1. 内容驱动的位置权重（避免固定位置嵌入）
    2. 多组基函数
    3. 门控机制
    """
    def __init__(self, d_model: int, num_basis: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_basis = num_basis
        self.d_model = d_model
        
        # ✨ FiLM 参数生成网络（更深）
        self.film_gen = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model * num_basis * 2)
        )
        
        # ✨ 内容驱动的位置权重生成器（基于序列内容动态决定）
        self.pos_proj = nn.Sequential(
            nn.Linear(d_model, num_basis),
            nn.Softmax(dim=-1)
        )
        
        # ✨ 门控机制
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
    def forward(self, seq: torch.Tensor, global_feat: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        seq: [B, L, D]
        global_feat: [B, D]
        Returns: [B, L, D]
        """
        B, L, D = seq.shape
        
        # 生成多组 FiLM 参数
        film_params = self.film_gen(global_feat)  # [B, num_basis * 2 * D]
        film_params = film_params.view(B, self.num_basis, 2, D)
        gamma_basis = film_params[:, :, 0, :]  # [B, num_basis, D]
        beta_basis = film_params[:, :, 1, :]   # [B, num_basis, D]
        
        # 基于序列内容生成位置权重
        pos_weights = self.pos_proj(seq)  # [B, L, num_basis]
        
        # 加权组合基函数
        gamma = torch.einsum('blk,bkd->bld', pos_weights, gamma_basis)
        beta = torch.einsum('blk,bkd->bld', pos_weights, beta_basis)
        
        # FiLM 调制
        modulated = gamma * seq + beta
        
        # 门控
        gate_values = self.gate(seq)
        enhanced = gate_values * modulated + (1 - gate_values) * seq
        
        if mask is not None:
            enhanced = enhanced * mask.unsqueeze(-1)
        
        return enhanced


# =========================
# ✨ 主融合模型（FiLM + 多层注入）
# =========================
class MultiModalProteinFusion(nn.Module):
    """
    改进架构：
    1. ESM2 为主干
    2. Foldseek 通过 FiLM 早期融合（保留调制能力）
    3. ProstT5 通过 FiLM 在每层注入（多次机会发挥作用）
    4. 更深的 ProstT5 编码器
    """
    def __init__(
        self,
        esm2_dim: int = 1280,
        foldseek_dim: int = 1024,
        prost5_dim: int = 768,
        d_model: int = 512,
        num_heads: int = 8,
        num_experts: int = 6,
        top_k_experts: int = 2,
        num_fusion_layers: int = 3,
        dropout: float = 0.2,
        num_film_basis: int = 4,
    ):
        super().__init__()
        
        # =========================
        # 1️⃣ 编码器
        # =========================
        
        # ESM2 序列编码器
        self.esm2_encoder = nn.Sequential(
            nn.Linear(esm2_dim, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Foldseek 结构编码器
        self.foldseek_encoder = nn.Sequential(
            nn.Linear(foldseek_dim, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # ✨ ProstT5 理化编码器（加深到 3 层）
        self.prost5_encoder = nn.Sequential(
            nn.Linear(prost5_dim, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        
        # =========================
        # 2️⃣ FiLM 增强器
        # =========================
        
        # 早期结构 FiLM（只在开头用一次）
        self.struct_film = EnhancedFiLM(d_model, num_basis=num_film_basis, dropout=dropout)
        
        # =========================
        # 3️⃣ 主干 Transformer 层 + 理化 FiLM
        # =========================
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "self_attn": CrossModalAttention(d_model, num_heads, dropout),
                "moe": MixtureOfExperts(d_model, num_experts, top_k_experts, dropout=dropout),
                "phys_film": EnhancedFiLM(d_model, num_basis=num_film_basis, dropout=dropout)  # ✨ 每层都有
            }) for _ in range(num_fusion_layers)
        ])
        
        self.final_ln = nn.LayerNorm(d_model)
        
        # ✨ 可学习的融合权重（给 ProstT5 更大的初始权重）
        self.struct_weight = nn.Parameter(torch.tensor(0.5))  # Foldseek 早期融合权重
        self.phys_weights = nn.Parameter(torch.ones(num_fusion_layers) * 0.8)  # ProstT5 每层权重

    def forward(
        self, 
        esm2_feat: torch.Tensor, 
        foldseek_feat: torch.Tensor, 
        prost5_feat: torch.Tensor, 
        temp: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            esm2_feat: [B, L, 1280]
            foldseek_feat: [B, 1024]
            prost5_feat: [B, 768]
            temp: MoE 温度
        
        Returns:
            seq: [B, L, D]
            total_lb: MoE 负载均衡损失
        """
        B, L, _ = esm2_feat.shape
        mask = (esm2_feat.abs().sum(dim=-1) > 0)
        
        # =========================
        # 阶段 1：编码 + 早期结构融合
        # =========================
        
        seq = self.esm2_encoder(esm2_feat)
        seq = self.pos_enc(seq)
        
        struct_global = self.foldseek_encoder(foldseek_feat)
        phys_global = self.prost5_encoder(prost5_feat)
        
        # ✨ 早期 Foldseek 融合（FiLM）
        struct_enhancement = self.struct_film(seq, struct_global, mask)
        struct_weight = torch.sigmoid(self.struct_weight)  # 限制在 [0, 1]
        seq = seq + struct_weight * struct_enhancement
        seq = seq * mask.unsqueeze(-1)
        
        # =========================
        # 阶段 2：主干处理 + 多层 ProstT5 注入
        # =========================
        
        total_lb = 0.0
        
        for i, layer in enumerate(self.layers):
            # Self-Attention
            seq = layer["self_attn"](seq, seq, seq, mask)
            seq = seq * mask.unsqueeze(-1)
            
            # MoE
            seq, lb = layer['moe'](seq, temp=temp)
            total_lb += lb
            
            # ✨ 晚期 ProstT5 融合（每层都通过 FiLM 注入）
            phys_enhancement = layer["phys_film"](seq, phys_global, mask)
            phys_weight = torch.sigmoid(self.phys_weights[i])  # 每层独立权重
            seq = seq + phys_weight * phys_enhancement
            seq = seq * mask.unsqueeze(-1)
        
        seq = self.final_ln(seq)
        
        return seq, total_lb


# =========================
# 使用示例
# =========================
if __name__ == "__main__":
    model = MultiModalProteinFusion(
        esm2_dim=1280,
        foldseek_dim=1024,
        prost5_dim=768,
        d_model=512,
        num_heads=8,
        num_experts=6,
        top_k_experts=2,
        num_fusion_layers=3,
        dropout=0.2,
        num_film_basis=4,
    )
    
    batch_size = 4
    seq_len = 100
    
    esm2_feat = torch.randn(batch_size, seq_len, 1280)
    foldseek_feat = torch.randn(batch_size, 1024)
    prost5_feat = torch.randn(batch_size, 768)
    
    output, lb_loss = model(esm2_feat, foldseek_feat, prost5_feat)
    
    print(f"Output shape: {output.shape}")
    print(f"MoE LB Loss: {lb_loss.item():.4f}")
    print(f"Struct weight: {torch.sigmoid(model.struct_weight).item():.3f}")
    print(f"Phys weights: {torch.sigmoid(model.phys_weights).detach().numpy()}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")