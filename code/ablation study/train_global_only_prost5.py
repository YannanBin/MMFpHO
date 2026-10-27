# train_global_only_prost5.py
# 消融实验：仅使用 ProstT5 全局特征
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.stats import pearsonr, spearmanr
import random
from metrics import calculate_regression_metrics, continuous_to_class, calculate_binary_metrics_ovr, calculate_multiclass_metrics
from sklearn.model_selection import train_test_split

# ------------------------------ 设置随机种子 ------------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# ------------------------------ 1. 加载特征 ------------------------------
def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

#prost5_dict = load_pkl("/data5/heyuhang/mywork/feature/features_512d_linear.pkl")
prost5_dict = load_pkl("/data5/heyuhang/mywork/feature/prost5_40d.pkl")
print("ProstT5 特征数:", len(prost5_dict))

# ------------------------------ 2. 全局特征专用模型 ------------------------------
class GlobalFeatureModel512(nn.Module):
    """
    全局特征专用回归模型
    无需 pooling，直接从固定维度向量预测
    """
    def __init__(self, input_dim=512):
        super().__init__()
        # 使用和你的回归头类似的结构
        self.regression_head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        """
        x: (B, input_dim) 全局特征
        Returns: (B,) 预测值
        """
        return self.regression_head(x).squeeze(-1)
    




class GlobalFeatureModel(nn.Module):
    """
    适合低维理化特征 (40d) 的轻量回归模型
    """
    def __init__(self, input_dim=40):
        super().__init__()

        self.regression_head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.regression_head(x).squeeze(-1)


# ------------------------------ 3. Dataset ------------------------------
class ProstT5OnlyDataset(Dataset):
    def __init__(self, df, prost5_dict, y_min=0.0, y_max=14.0):
        self.samples = []
        for acc, label in zip(df["ID"], df["label"]):
            if acc in prost5_dict:
                self.samples.append((acc, float(label)))
        
        self.prost5_dict = prost5_dict
        self.y_min = y_min
        self.y_max = y_max
        print(f"有效样本数: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        acc, label = self.samples[idx]
        
        # ProstT5 特征 (处理 NaN)
        prost5_feat = self.prost5_dict[acc]
        prost5_feat = np.nan_to_num(prost5_feat, nan=0.0)
        prost5_feat = torch.tensor(prost5_feat, dtype=torch.float32)
        
        if torch.isnan(prost5_feat).any():
            print(f"Warning: NaN detected in ProstT5 for ID={acc}")
        
        # 标签归一化到 [0,1]
        y = torch.tensor((label - self.y_min) / (self.y_max - self.y_min), dtype=torch.float32)
        
        return prost5_feat, y

# ------------------------------ 4. EarlyStopping ------------------------------
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def step(self, score):
        if self.best_score is None:
            self.best_score = score
            return True
        elif score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False

# ------------------------------ 5. 数据加载 ------------------------------
df = pd.read_csv("/data5/heyuhang/mywork/mydata/label.csv")
trainval_df = df[df["Split"].isin(["Training", "Validation"])]

# 随机划分 90% 训练 / 10% 验证
train_df, val_df = train_test_split(
    trainval_df,
    test_size=0.1,
    random_state=42,
    shuffle=True
)

batch_size = 32

train_dataset = ProstT5OnlyDataset(train_df, prost5_dict)
val_dataset = ProstT5OnlyDataset(val_df, prost5_dict)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
)

# ------------------------------ 6. 模型训练 ------------------------------
model = GlobalFeatureModel(input_dim=40)
early_stopper = EarlyStopping(patience=20, min_delta=1e-4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

num_epochs = 100
y_min, y_max = 0.0, 14.0

print(f"\n{'='*60}")
print(f"开始训练: 仅使用 ProstT5 全局特征")
print(f"模型结构: {sum(p.numel() for p in model.parameters())/1e6:.2f}M 参数")
print(f"{'='*60}\n")

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for prost5_feat, y in train_loader:
        prost5_feat = prost5_feat.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        pred_raw = model(prost5_feat)
        pred = torch.sigmoid(pred_raw)

        loss = F.mse_loss(pred, y)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        train_loss += loss.item() * prost5_feat.size(0)

    train_loss /= len(train_dataset)

    # -------- Validation --------
    model.eval()
    all_y, all_pred = [], []

    with torch.no_grad():
        for prost5_feat, y in val_loader:
            prost5_feat = prost5_feat.to(device)
            y = y.to(device)

            pred_raw = model(prost5_feat)
            pred = torch.sigmoid(pred_raw)
            
            all_y.append(y.cpu().numpy())
            all_pred.append(pred.cpu().numpy())

    all_y = np.concatenate(all_y, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)

    # 还原回原始 pH
    all_y_ph = all_y * (y_max - y_min) + y_min
    all_pred_ph = all_pred * (y_max - y_min) + y_min

    # 转分类
    true_classes = continuous_to_class(all_y_ph)
    pred_classes = continuous_to_class(all_pred_ph)

    # 计算指标
    reg_metrics = calculate_regression_metrics(all_y_ph, all_pred_ph)
    binary_metrics = calculate_binary_metrics_ovr(true_classes, pred_classes)

    # 打印
    print(
        f"Epoch [{epoch+1:3d}/{num_epochs}] "
        f"Loss: {train_loss:.4f} | "
        f"R²: {reg_metrics['R2']:6.4f} "
        f"RMSE: {reg_metrics['RMSE']:.4f} "
        f"Pearson: {reg_metrics['Pearson']:.4f} "
        f"Spearman: {reg_metrics['Spearman']:.4f} | "
        f"ACC: {binary_metrics['Accuracy']:.4f} "
        f"MCC: {binary_metrics['MCC']:.4f} "
        f"F1: {binary_metrics['F1']:.4f} "
        f"AUC: {binary_metrics['AUC']:.4f}"
    )

    # Learning rate scheduling
    scheduler.step(reg_metrics['R2'])

    # Early Stopping
    improved = early_stopper.step(reg_metrics['R2'])
    if improved:
        torch.save(model.state_dict(), "/data5/heyuhang/mywork/MOE/best_model_prost5_only40.pt")
        print(f"  ✔ Best model saved (R²={reg_metrics['R2']:.4f})")

    if early_stopper.early_stop:
        print(f"\n⛔ Early stopping at epoch {epoch+1}")
        break

print(f"\n{'='*60}")
print(f"训练完成: 仅使用 ProstT5")
print(f"最佳 R²: {early_stopper.best_score:.4f}")
print(f"{'='*60}\n")