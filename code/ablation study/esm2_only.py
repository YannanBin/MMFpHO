# train_esm2_only_token512_ablation.py
# ESM2 only baseline
# STRONG ABLATION: token-level 1280 → 512 before pooling

import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

from metrics import (
    calculate_regression_metrics,
    continuous_to_class,
    calculate_binary_metrics_ovr,
    calculate_multiclass_metrics
)

# ------------------------------ Random Seed ------------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# ------------------------------ Utils ------------------------------
def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# ------------------------------ Load ESM2 features ------------------------------
esm_dict = load_pkl("/data5/heyuhang/mywork/feature/trainesm2.pkl")
print("ESM2 samples:", len(esm_dict))

# ------------------------------ Dataset ------------------------------
class ProteinDatasetESM2Only(Dataset):
    def __init__(self, df, esm_dict, y_min=0.0, y_max=14.0):
        self.samples = []
        for acc, label in zip(df["ID"], df["label"]):
            if acc in esm_dict:
                self.samples.append((acc, float(label)))

        self.esm_dict = esm_dict
        self.y_min = y_min
        self.y_max = y_max
        print(f"有效样本数: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        acc, label = self.samples[idx]

        esm = torch.tensor(self.esm_dict[acc], dtype=torch.float32)
        y = torch.tensor(
            (label - self.y_min) / (self.y_max - self.y_min),
            dtype=torch.float32
        )
        return esm, y


def collate_fn(batch):
    esm_list, y_list = zip(*batch)
    esm_pad = pad_sequence(esm_list, batch_first=True)
    y = torch.stack(y_list).unsqueeze(1)
    return esm_pad, y

# ------------------------------ Model ------------------------------
class MeanPooling(nn.Module):
    def forward(self, x, mask):
        mask = mask.unsqueeze(-1)
        return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)


class ESM2RegressionModel(nn.Module):
    def __init__(self, esm_dim=1280):
        super().__init__()

        # 🔥 token级压缩
        self.token_proj = nn.Linear(esm_dim, 256)   # 1280 → 64

        self.regression_head = nn.Sequential(
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, esm_feat):
        # (B, L, 1280) → (B, L, 64)
        x = self.token_proj(esm_feat)

        # mean pooling
        x = x.mean(dim=1)   # (B, 64)

        return self.regression_head(x).squeeze(-1)
# ------------------------------ EarlyStopping ------------------------------
class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def step(self, score):
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False

# ------------------------------ Data Split ------------------------------
df = pd.read_csv("/data5/heyuhang/mywork/mydata/label.csv")
trainval_df = df[df["Split"].isin(["Training", "Validation"])]

train_df, val_df = train_test_split(
    trainval_df,
    test_size=0.1,
    random_state=42,
    shuffle=True
)

batch_size = 8

train_dataset = ProteinDatasetESM2Only(train_df, esm_dict)
val_dataset = ProteinDatasetESM2Only(val_df, esm_dict)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn
)

# ------------------------------ Training Config ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ESM2RegressionModel().to(device)

optimizer = optim.AdamW(model.parameters(), lr=5e-4)
num_epochs = 100
early_stopper = EarlyStopping(patience=20)

y_min, y_max = 0.0, 14.0

print("\n" + "=" * 60)
print("Start Training: ESM2 only (token-level 512 ablation)")
print("=" * 60 + "\n")

# ------------------------------ Training Loop ------------------------------
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for esm, y in train_loader:
        esm = esm.to(device)
        y = y.to(device).squeeze(1)

        pred_raw = model(esm)
        pred = torch.sigmoid(pred_raw)

        loss = F.mse_loss(pred, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        train_loss += loss.item() * esm.size(0)

    train_loss /= len(train_dataset)

    # ---------------- Validation ----------------
    model.eval()
    all_y, all_pred = [], []

    with torch.no_grad():
        for esm, y in val_loader:
            esm = esm.to(device)
            y = y.to(device).squeeze(1)

            pred_raw = model(esm)
            pred = torch.sigmoid(pred_raw)

            all_y.append(y.cpu().numpy())
            all_pred.append(pred.cpu().numpy())

    all_y = np.concatenate(all_y)
    all_pred = np.concatenate(all_pred)

    all_y_ph = all_y * (y_max - y_min) + y_min
    all_pred_ph = all_pred * (y_max - y_min) + y_min

    true_classes = continuous_to_class(all_y_ph)
    pred_classes = continuous_to_class(all_pred_ph)

    reg_metrics = calculate_regression_metrics(all_y_ph, all_pred_ph)
    binary_metrics = calculate_binary_metrics_ovr(true_classes, pred_classes)

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

    improved = early_stopper.step(reg_metrics["Spearman"])
    if improved:
        torch.save(
            model.state_dict(),
            "/data5/heyuhang/mywork/MOE/best_model_esm2_only.pt"
        )
        print("  ✔ Best model saved")

    if early_stopper.early_stop:
        print(f"⛔ Early stopping at epoch {epoch+1}")
        break

print("\nTraining finished: strongest ESM2-only ablation\n")
