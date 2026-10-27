# infer_esm2_only_token512_ablation.py
# Batch inference: ESM2 only (token-level 512 ablation)

import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np

# ------------------------------ Config ------------------------------
MODEL_PATH = "/data5/heyuhang/mywork/MOE/best_model_esm2_only.pt"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Y_MIN, Y_MAX = 0.0, 14.0

print("\n" + "=" * 60)
print("Using model: ESM2 only (token-level 512 ablation)")
print(f"Model path: {MODEL_PATH}")
print("=" * 60 + "\n")

# ------------------------------ Load ESM2 features ------------------------------
def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

esm_dict = load_pkl("/data5/heyuhang/mywork/mydata/phtest/testesm2.pkl")
print("ESM2 feature count:", len(esm_dict))

# ------------------------------ Dataset ------------------------------
class ProteinDatasetESM2Only(Dataset):
    def __init__(self, ids, esm_dict):
        self.samples = [acc for acc in ids if acc in esm_dict]
        self.esm_dict = esm_dict
        print(f"有效样本数: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        acc = self.samples[idx]
        esm = torch.tensor(self.esm_dict[acc], dtype=torch.float32)
        return acc, esm


def collate_fn(batch):
    ids, esm_list = zip(*batch)
    esm_pad = pad_sequence(esm_list, batch_first=True)
    return ids, esm_pad

# ------------------------------ Model (MUST match training) ------------------------------
class MeanPooling(nn.Module):
    def forward(self, x, mask):
        mask = mask.unsqueeze(-1)
        return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)


class ESM2RegressionModel(nn.Module):
    """
    Token-level 1280 → 512 projection BEFORE pooling
    """
    def __init__(self, esm_dim=1280, proj_dim=512):
        super().__init__()

        self.token_proj = nn.Linear(esm_dim, proj_dim)
        self.pooling = MeanPooling()

        self.regression_head = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(proj_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, esm_feat):
        # (B, L, 1280) → (B, L, 512)
        x = self.token_proj(esm_feat)

        mask = (x.abs().sum(dim=-1) > 0).float()
        pooled = self.pooling(x, mask)

        return self.regression_head(pooled).squeeze(-1)

# ------------------------------ Load model ------------------------------
model = ESM2RegressionModel().to(DEVICE)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()

print("Model loaded successfully\n")

# ------------------------------ Inference ------------------------------
def inference(ids_list, batch_size=BATCH_SIZE):
    dataset = ProteinDatasetESM2Only(ids_list, esm_dict)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    results = []

    with torch.no_grad():
        for ids, esm in loader:
            esm = esm.to(DEVICE)

            pred_raw = model(esm)
            pred = torch.sigmoid(pred_raw)
            pred_ph = pred.cpu().numpy() * (Y_MAX - Y_MIN) + Y_MIN

            for acc, val in zip(ids, pred_ph):
                results.append((acc, float(val)))

    return results

# ------------------------------ Main ------------------------------
if __name__ == "__main__":
    df_input = pd.read_csv(
        "/data5/heyuhang/mywork/mydata/phtest/test_unique.csv"
    )
    ids_list = df_input["Accession"].tolist()

    print(f"Start inference on {len(ids_list)} samples...\n")
    results = inference(ids_list)

    output_file = "/data5/heyuhang/mywork/MOE/predictions_esm2_only_token512_ablation.csv"
    df_out = pd.DataFrame(results, columns=["Accession", "Predicted_pH"])
    df_out.to_csv(output_file, index=False)

    print("\n" + "=" * 60)
    print("Inference finished!")
    print("=" * 60 + "\n")
