# train_sigmoid.py
#sigmod
import pickle
import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from feature9 import CrossModalFusionWithMoE
from model import ProteinRegressionModel
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

esm_dict = load_pkl("/data5/heyuhang/mywork/feature/trainesm2.pkl")
struct_dict = load_pkl("/data5/heyuhang/mywork/feature/finalstructure.pkl")
prost_dict = load_pkl("/data5/heyuhang/mywork/feature/features_512d_linear.pkl")

print("ESM2:", len(esm_dict))
print("Foldseek:", len(struct_dict))
print("ProstT5:", len(prost_dict))

# ------------------------------ Dataset ------------------------------
class ProteinDataset(Dataset):
    def __init__(self, df, esm_dict, struct_dict, prost_dict, y_min=0.0, y_max=14.0):
        self.samples = []
        for acc, label in zip(df["ID"], df["label"]):
            if acc in esm_dict and acc in struct_dict and acc in prost_dict:
                self.samples.append((acc, float(label)))
        self.esm_dict = esm_dict
        self.struct_dict = struct_dict
        self.prost_dict = prost_dict
        self.y_min = y_min
        self.y_max = y_max
        print(f"有效样本数: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        acc, label = self.samples[idx]
        esm = torch.tensor(self.esm_dict[acc], dtype=torch.float32)
        struct = torch.tensor(self.struct_dict[acc], dtype=torch.float32)
        prost_np = np.nan_to_num(self.prost_dict[acc], nan=0.0)
        prost = torch.tensor(prost_np, dtype=torch.float32)
        if torch.isnan(prost).any():
            print(f"Warning: NaN detected in ProstT5 for ID={acc}")
        # 标签归一化到 [0,1]
        y = torch.tensor((label - self.y_min) / (self.y_max - self.y_min), dtype=torch.float32)
        return esm, struct, prost, y

def collate_fn(batch):
    esm_list, struct_list, prost_list, y_list = zip(*batch)
    esm_pad = pad_sequence(esm_list, batch_first=True)
    struct = torch.stack(struct_list)
    prost = torch.stack(prost_list)
    y = torch.stack(y_list).unsqueeze(1)
    return esm_pad, struct, prost, y

# ------------------------------ EarlyStopping ------------------------------
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

# ------------------------------ 2. 数据加载 ------------------------------
# ------------------------------ 2. 数据加载 ------------------------------
df = pd.read_csv("/data5/heyuhang/mywork/mydata/label.csv")
trainval_df = df[df["Split"].isin(["Training", "Validation"])]

# 随机划分 90% 训练 / 10% 验证
train_df, val_df = train_test_split(
    trainval_df,
    test_size=0.1,
    random_state=42,
    shuffle=True
)

batch_size = 8

train_dataset = ProteinDataset(train_df, esm_dict, struct_dict, prost_dict)
val_dataset = ProteinDataset(val_df, esm_dict, struct_dict, prost_dict)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    collate_fn=collate_fn, num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
)
# ------------------------------ 3. 模型 ------------------------------
feature_extractor = CrossModalFusionWithMoE(
    esm2_dim=1280,
    foldseek_dim=train_dataset[0][1].shape[0],
    prost5_dim=512,
    d_model=512,
    num_heads=8,
    num_experts=6,
    top_k_experts=2,
    num_fusion_layers=2,
    dropout=0.2
)
model = ProteinRegressionModel(feature_extractor, d_model=512)
early_stopper = EarlyStopping(patience=30, min_delta=1e-4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=5e-5)
num_epochs = 100

# ------------------------------ 4. 训练循环（梯度累计 + MoE gating warmup） ------------------------------
accum_steps = 4
warmup_epochs = 3
y_min, y_max = 0.0, 14.0  # pH 范围

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    optimizer.zero_grad()

    # ===== Warmup 模态 scale =====
    model.feature_extractor.modality_scale.requires_grad = epoch >= warmup_epochs

    for step, (esm, struct, prost, y) in enumerate(train_loader):
        esm = esm.to(device)
        struct = struct.to(device)
        prost = prost.to(device)
        y = y.to(device).squeeze(1)

        # ===== forward =====
        pred_raw, lb_loss = model(esm, struct, prost, temp=2.0 if epoch < warmup_epochs else 1.0)
        pred = torch.sigmoid(pred_raw)  # Sigmoid 输出 [0,1]

        #task_loss = F.mse_loss(pred, y)
        task_loss = F.huber_loss(pred, y, delta=1.0)
        loss = task_loss + 0.05 * lb_loss
        loss = loss / accum_steps
        loss.backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            optimizer.zero_grad()

        train_loss += loss.item() * esm.size(0) * accum_steps

    train_loss /= len(train_dataset)

    # -------- Validation --------
    model.eval()
    all_y, all_pred = [], []

    with torch.no_grad():
        for esm, struct, prost, y in val_loader:
            esm = esm.to(device)
            struct = struct.to(device)
            prost = prost.to(device)
            y = y.to(device).squeeze(1)

            pred_raw, _ = model(esm, struct, prost)
            pred = torch.sigmoid(pred_raw)  # Sigmoid 输出
            all_y.append(y.cpu().numpy())
            all_pred.append(pred.cpu().numpy())

    all_y = np.concatenate(all_y, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)

    # ===== 还原回原始 pH =====
    all_y_ph = all_y * (y_max - y_min) + y_min
    all_pred_ph = all_pred * (y_max - y_min) + y_min

    # ===== 转分类 =====
    true_classes = continuous_to_class(all_y_ph)
    pred_classes = continuous_to_class(all_pred_ph)

    # ===== 计算指标 =====
    reg_metrics = calculate_regression_metrics(all_y_ph, all_pred_ph)
    binary_metrics = calculate_binary_metrics_ovr(true_classes, pred_classes)
    multiclass_metrics = calculate_multiclass_metrics(true_classes, pred_classes)

    # ===== 打印 =====
    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"R2: {reg_metrics['R2']:.4f} "
        f"RMSE: {reg_metrics['RMSE']:.4f} "
        f"Pearson: {reg_metrics['Pearson']:.4f} "
        f"Spearman: {reg_metrics['Spearman']:.4f} "
        f"ACC: {binary_metrics['Accuracy']:.4f} "
        f"MCC: {binary_metrics['MCC']:.4f} "
        f"F1: {binary_metrics['F1']:.4f} "
        f"AUC: {binary_metrics['AUC']:.4f} "
    )

    # ===== Early Stopping =====
    improved = early_stopper.step(reg_metrics['Spearman'])
    if improved:
        torch.save(model.state_dict(), "/data5/heyuhang/mywork/MOE/best_model_sigmoid10.pt")
        print("  ✔ Best model saved")

    if early_stopper.early_stop:
        print(f"⛔ Early stopping at epoch {epoch+1}")
        break
