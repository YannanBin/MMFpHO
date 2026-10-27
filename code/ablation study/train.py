# train_ablation.py
import pickle
import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import random
from sklearn.model_selection import train_test_split

from feature10 import SequenceDominantFusion
from model import ProteinRegressionModel
from metrics import calculate_regression_metrics, continuous_to_class, calculate_binary_metrics_ovr, calculate_multiclass_metrics

# ------------------------------ 设置随机种子 ------------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# ------------------------------ 训练/验证循环 ------------------------------
def train_one_epoch(model, loader, optimizer, accum_steps=4, temp=1.0, use_struct=True, use_prost=True):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    for step, (esm, struct, prost, y) in enumerate(loader):
        esm, struct, prost, y = esm.to(device), struct.to(device), prost.to(device), y.to(device).squeeze(1)

        # ==== Ablation 消融 ====
        if not use_struct:
            struct = torch.zeros_like(struct)
        if not use_prost:
            prost = torch.zeros_like(prost)

        pred_raw, lb_loss = model(esm, struct, prost, temp=temp)
        pred = torch.sigmoid(pred_raw)
        task_loss = F.mse_loss(pred, y)
        loss = task_loss + 0.05 * lb_loss
        loss = loss / accum_steps
        loss.backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * esm.size(0) * accum_steps

    return total_loss / len(loader.dataset)

def validate(model, loader, use_struct=True, use_prost=True, y_min=0.0, y_max=14.0):
    model.eval()
    all_y, all_pred = [], []

    with torch.no_grad():
        for esm, struct, prost, y in loader:
            esm, struct, prost, y = esm.to(device), struct.to(device), prost.to(device), y.to(device).squeeze(1)
            if not use_struct:
                struct = torch.zeros_like(struct)
            if not use_prost:
                prost = torch.zeros_like(prost)

            pred_raw, _ = model(esm, struct, prost)
            pred = torch.sigmoid(pred_raw)
            all_y.append(y.cpu().numpy())
            all_pred.append(pred.cpu().numpy())

    all_y = np.concatenate(all_y)
    all_pred = np.concatenate(all_pred)
    all_y_ph = all_y * (y_max - y_min) + y_min
    all_pred_ph = all_pred * (y_max - y_min) + y_min
    reg_metrics = calculate_regression_metrics(all_y_ph, all_pred_ph)
    true_classes = continuous_to_class(all_y_ph)
    pred_classes = continuous_to_class(all_pred_ph)
    binary_metrics = calculate_binary_metrics_ovr(true_classes, pred_classes)
    multiclass_metrics = calculate_multiclass_metrics(true_classes, pred_classes)
    return reg_metrics, binary_metrics, multiclass_metrics

# ------------------------------ Ablation 实验 ------------------------------
if __name__ == "__main__":
    df = pd.read_csv("/data5/heyuhang/mywork/mydata/label.csv")
    trainval_df = df[df["Split"].isin(["Training","Validation"])]
    train_df, val_df = train_test_split(trainval_df, test_size=0.1, random_state=42, shuffle=True)

    batch_size = 8
    train_loader = DataLoader(ProteinDataset(train_df, esm_dict, struct_dict, prost_dict),
                              batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(ProteinDataset(val_df, esm_dict, struct_dict, prost_dict),
                            batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Ablation 设置列表
    ablation_settings = [
        ("Full", True, True),
        ("No ProstT5", True, False),
        ("No Struct", False, True),
        ("Only ESM2", False, False)
    ]

    results = []

    for name, use_struct, use_prost in ablation_settings:
        print(f"\n===== Training Ablation Setting: {name} =====")
        feature_extractor = SequenceDominantFusion(
            esm2_dim=1280, foldseek_dim=1024, prost5_dim=512,
            d_model=512, num_heads=8, num_experts=6,
            top_k_experts=2, num_fusion_layers=2, dropout=0.2
        )
        model = ProteinRegressionModel(feature_extractor, d_model=512).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=5e-5)
        early_stopper = EarlyStopping(patience=15, min_delta=1e-4)
        num_epochs = 50
        warmup_epochs = 3
        # ===== Training 循环 =====
        for epoch in range(num_epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer,
                                        accum_steps=4,
                                        temp=2.0 if epoch < warmup_epochs else 1.0,
                                        use_struct=use_struct,
                                        use_prost=use_prost)

            # ===== Validation =====
            reg_metrics, binary_metrics, multiclass_metrics = validate(model, val_loader,
                                                                    use_struct=use_struct,
                                                                    use_prost=use_prost)

            print(
                f"Epoch [{epoch+1}/{num_epochs}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"R2: {reg_metrics['R2']:.4f} | RMSE: {reg_metrics['RMSE']:.4f} | "
                f"Pearson: {reg_metrics['Pearson']:.4f} | Spearman: {reg_metrics['Spearman']:.4f} | "
                f"ACC: {binary_metrics['Accuracy']:.4f} | MCC: {binary_metrics['MCC']:.4f} | "
                f"F1: {binary_metrics['F1']:.4f} | AUC: {binary_metrics['AUC']:.4f}"
            )


            # Early stopping
            improved = early_stopper.step(reg_metrics['R2'])
            if improved:
                torch.save(model.state_dict(), f"best_model_{name.replace(' ','_')}.pt")
            if early_stopper.early_stop:
                print(f"⛔ Early stopping at epoch {epoch+1}")
                break

        # 最终验证结果
        reg_metrics, binary_metrics, multiclass_metrics = validate(model, val_loader,
                                                                   use_struct=use_struct,
                                                                   use_prost=use_prost)
        results.append((name, reg_metrics, binary_metrics, multiclass_metrics))

    # -------------------- 打印对比表 --------------------
    print("\n===== Ablation Results =====")
    for name, reg, bin_m, multi_m in results:
        print(f"{name}: R2={reg['R2']:.4f}, RMSE={reg['RMSE']:.4f}, Pearson={reg['Pearson']:.4f}")
