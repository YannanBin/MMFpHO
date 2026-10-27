# model.py
import torch
import torch.nn as nn


class MeanPooling(nn.Module):
    """Mask-aware mean pooling"""
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1)  # (B, L, 1)
        return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)


class AttentionPooling(nn.Module):
    """Mask-aware attention pooling"""
    def __init__(self, d_model: int):
        super().__init__()
        self.attn_score = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.attn_score(x).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(scores, dim=-1)
        pooled = torch.einsum("bl,bld->bd", attn_weights, x)
        return pooled


class HybridPooling(nn.Module):
    """Mean + Attention hybrid pooling"""
    def __init__(self, d_model: int):
        super().__init__()
        self.attn_score = nn.Linear(d_model, 1)
        self.alpha = nn.Parameter(torch.tensor(-1.5))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.float()
        mean_pool = (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1e-6)
        scores = self.attn_score(x).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn_pool = torch.einsum("bl,bld->bd", attn, x)
        alpha = torch.sigmoid(self.alpha)
        pooled = (1 - alpha) * mean_pool + alpha * attn_pool
        return pooled


class ProteinRegressionModelFiLM(nn.Module):
    """
    兼容 MoE 版本和 FiLM 版本的通用回归模型
    """
    def __init__(self, feature_extractor, d_model: int = 512):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.pooling =HybridPooling()  # 可以切换 AttentionPooling 或 HybridPooling
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, esm2_feat, foldseek_feat, prost5_feat):
        """
        esm2_feat    : (B, L, 1280) padded
        foldseek_feat: (B, L, D_struct)
        prost5_feat  : (B, 512)
        """
        # 1️⃣ 生成 padding mask
        mask = (esm2_feat.abs().sum(dim=-1) > 0).float()  # (B, L)

        # 2️⃣ 融合特征
        fused = self.feature_extractor(esm2_feat, foldseek_feat, prost5_feat)

        # 兼容 MoE 版本（返回 tuple）和 FiLM 版本（返回 tensor）
        if isinstance(fused, tuple):
            fused_feat, aux_loss = fused
        else:
            fused_feat = fused
            aux_loss = torch.tensor(0.0, device=fused_feat.device)

        # 3️⃣ Mask-aware pooling
        pooled = self.pooling(fused_feat, mask)  # (B, d_model)

        # 4️⃣ 回归
        prediction = self.regression_head(pooled).squeeze(-1)

        return prediction, aux_loss







class ProteinRegressionModel(nn.Module):
    """
    通用蛋白质回归模型，兼容 MoE 版本和 FiLM 版本
    支持 soft routing 温度传入
    """
    def __init__(self, feature_extractor, d_model: int = 512):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.pooling = HybridPooling(d_model)  # 或者使用 AttentionPooling/HybridPooling
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, esm2_feat, foldseek_feat, prost5_feat, temp: float = 1.0):
        """
        esm2_feat    : (B, L, 1280) padded
        foldseek_feat: (B, D_struct)
        prost5_feat  : (B, 512)
        temp         : softmax 温度，控制 MoE soft routing
        """
        # 1️⃣ padding mask
        mask = (esm2_feat.abs().sum(dim=-1) > 0).float()  # (B,L)

        # 2️⃣ 融合特征
        fused = self.feature_extractor(esm2_feat, foldseek_feat, prost5_feat, temp=temp)

        # 兼容 MoE 返回 tuple 或 FiLM 返回 tensor
        if isinstance(fused, tuple):
            fused_feat, aux_loss = fused
        else:
            fused_feat = fused
            aux_loss = torch.tensor(0.0, device=fused_feat.device)

        # 3️⃣ Mask-aware pooling
        pooled = self.pooling(fused_feat, mask)  # (B, d_model)

        # 4️⃣ 回归输出
        prediction = self.regression_head(pooled).squeeze(-1)

        return prediction, aux_loss
    


class ProteinRegressionModel1(nn.Module):
    """
    更新后的蛋白质回归模型包装器
    适配 Noisy Top-k MoE 和 Sigmoid 训练逻辑
    """
    def __init__(self, feature_extractor, d_model: int = 512):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.pooling = MeanPooling()  
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, esm2_feat, foldseek_feat, prost5_feat, temp: float = None):
        """
        temp: 控制 MoE 路由的平滑度。
              训练阶段：传入 1.2 增加探索；
              测试阶段：传入 1.0 或 None 使用默认硬路由。
        """
        # 1️⃣ 计算 Mask (B, L)
        # 只要特征在最后一个维度不全为 0，即视为有效残基
        mask = (esm2_feat.abs().sum(dim=-1) > 0).float()

        # 2️⃣ 提取特征
        # 将 temp 准确透传给内部的 MixtureOfExperts 模块
        fused = self.feature_extractor(esm2_feat, foldseek_feat, prost5_feat, temp=temp)

        # 兼容 MoE 的辅助损失返回
        if isinstance(fused, tuple):
            fused_feat, aux_loss = fused
        else:
            fused_feat = fused
            aux_loss = torch.tensor(0.0, device=fused_feat.device)

        # 3️⃣ Mask-aware pooling 
        # 将 (B, L, D) 聚合为 (B, D)
        pooled = self.pooling(fused_feat, mask)

        # 4️⃣ 回归输出 (B,)
        # 输出的是 Raw Logits，后续在 train_sigmoid.py 中通过 torch.sigmoid 映射到 [0, 1]
        prediction = self.regression_head(pooled).squeeze(-1)

        return prediction, aux_loss