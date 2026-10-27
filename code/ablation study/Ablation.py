# train_ablation_single.py
import pickle
import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

from feature12 import MultiModalProteinFusion
from model import ProteinRegressionModel
from metrics import calculate_regression_metrics, continuous_to_class, calculate_binary_metrics_ovr, calculate_multiclass_metrics

# ------------------------------ 随机种子 ------------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# ------------------------------ 加载特征 ------------------------------
def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

esm_dict = load_pkl("/data5/heyuhang/mywork/feature/trainesm2.pkl")
struct_dict = load_pkl("/data5/heyuhang/mywork/feature/finalstructure.pkl")
prost_dict = load_pkl("/data5/heyuhang/mywork/GO/biobert/biobert_embeddings1.pkl")

# ------------------------------ Dataset ------------------------------
class ProteinDataset(Dataset):
    def __init__(self, df, esm_dict, struct_dict, prost_dict, y_min=0.0, y_max=14.0):
        self.samples = [(acc, float(label)) for acc, label in zip(df["ID"], df["label"])
                        if acc in esm_dict and acc in struct_dict and acc in prost_dict]
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

# ------------------------------ 配置消融 ------------------------------
cfg_flags = {
    "zero_struct": True,  # True 时消融 Foldseek
    "zero_phys": False,    # True 时消融 ProstT5
    # 如果需要消融 FiLM 或 MoE，可在 MultiModalProteinFusion.forward 中屏蔽
}

# ------------------------------ 训练 ------------------------------
if __name__ == "__main__":
    df = pd.read_csv("/data5/heyuhang/mywork/mydata/label.csv")
    trainval_df = df[df["Split"].isin(["Training", "Validation"])]
    train_df, val_df = train_test_split(trainval_df, test_size=0.1, random_state=42, shuffle=True)

    batch_size = 8
    train_dataset = ProteinDataset(train_df, esm_dict, struct_dict, prost_dict)
    val_dataset = ProteinDataset(val_df, esm_dict, struct_dict, prost_dict)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # ------------------------------ 模型 ------------------------------
    feature_extractor = MultiModalProteinFusion(
        esm2_dim=1280,
        foldseek_dim=1024,
        prost5_dim=40,
        d_model=512,
        num_heads=8,
        num_experts=6,
        top_k_experts=2,
        num_fusion_layers=2,
        dropout=0.2,
        num_film_basis=4,
    )
    model = ProteinRegressionModel(feature_extractor, d_model=512)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    early_stopper = EarlyStopping(patience=30, min_delta=1e-4)
    num_epochs = 100
    accum_steps = 4
    warmup_epochs = 3
    y_min, y_max = 0.0, 14.0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        for step, (esm, struct, prost, y) in enumerate(train_loader):
            esm, struct, prost, y = esm.to(device), struct.to(device), prost.to(device), y.to(device).squeeze(1)

            # -------------------- 消融 --------------------
            if cfg_flags["zero_struct"]:
                struct = torch.zeros_like(struct)
            if cfg_flags["zero_phys"]:
                prost = torch.zeros_like(prost)

            pred_raw, lb_loss = model(esm, struct, prost, temp=2.0 if epoch < warmup_epochs else 1.0)
            pred = torch.sigmoid(pred_raw)
            task_loss = F.mse_loss(pred, y)
            loss = task_loss + 0.05 * lb_loss
            loss = loss / accum_steps
            loss.backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * esm.size(0) * accum_steps

        train_loss /= len(train_dataset)

        # ---------- 验证 ----------
        model.eval()
        all_y, all_pred = [], []
        val_loss = 0.0
        with torch.no_grad():
            for esm, struct, prost, y in val_loader:
                esm, struct, prost, y = esm.to(device), struct.to(device), prost.to(device), y.to(device).squeeze(1)

                if cfg_flags["zero_struct"]:
                    struct = torch.zeros_like(struct)
                if cfg_flags["zero_phys"]:
                    prost = torch.zeros_like(prost)

                pred_raw, lb_loss = model(esm, struct, prost)
                pred = torch.sigmoid(pred_raw)
                task_loss = F.mse_loss(pred, y)
                val_loss += task_loss.item() * esm.size(0)
                all_y.append(y.cpu().numpy())
                all_pred.append(pred.cpu().numpy())

        val_loss /= len(val_dataset)
        all_y_ph = np.concatenate(all_y, axis=0) * (y_max - y_min) + y_min
        all_pred_ph = np.concatenate(all_pred, axis=0) * (y_max - y_min) + y_min

        true_classes = continuous_to_class(all_y_ph)
        pred_classes = continuous_to_class(all_pred_ph)

        reg_metrics = calculate_regression_metrics(all_y_ph, all_pred_ph)
        binary_metrics = calculate_binary_metrics_ovr(true_classes, pred_classes)
        multiclass_metrics = calculate_multiclass_metrics(true_classes, pred_classes)

        print(f"Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}, R2={reg_metrics['R2']:.4f}")

        improved = early_stopper.step(reg_metrics['R2'])
        if improved:
            save_path = "/data5/heyuhang/mywork/MOE/best_model_no_foldseek.pt"
            torch.save(model.state_dict(), save_path)
            print(f"  ✔ Best model saved to {save_path}")

        if early_stopper.early_stop:
            print(f"⛔ Early stopping at epoch {epoch+1}")
            break
