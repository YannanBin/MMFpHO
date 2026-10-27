# infer_sigmoid.py
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
from feature9 import CrossModalFusionWithMoE
from model import ProteinRegressionModel

# ------------------------------ 设置 ------------------------------
MODEL_PATH = "/data5/heyuhang/mywork/MOE/best_model_sigmoid9.pt"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Y_MIN, Y_MAX = 0.0, 14.0  # pH 范围

# ------------------------------ 加载特征 ------------------------------
def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

esm_dict = load_pkl("/data5/heyuhang/mywork/mydata/phtest/testesm2.pkl")
struct_dict = load_pkl("/data5/heyuhang/mywork/mydata/phtest/testfinalstructure.pkl")
prost_dict = load_pkl("/data5/heyuhang/mywork/mydata/phtest/features_512d_liner1.pkl")

print("ESM2:", len(esm_dict))
print("Foldseek:", len(struct_dict))
print("ProstT5:", len(prost_dict))

# ------------------------------ Dataset ------------------------------
class ProteinDataset(Dataset):
    def __init__(self, ids, esm_dict, struct_dict, prost_dict):
        self.samples = [acc for acc in ids if acc in esm_dict and acc in struct_dict and acc in prost_dict]
        self.esm_dict = esm_dict
        self.struct_dict = struct_dict
        self.prost_dict = prost_dict
        print(f"有效样本数: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        acc = self.samples[idx]
        esm = torch.tensor(self.esm_dict[acc], dtype=torch.float32)
        struct = torch.tensor(self.struct_dict[acc], dtype=torch.float32)
        prost_np = np.nan_to_num(self.prost_dict[acc], nan=0.0)
        prost = torch.tensor(prost_np, dtype=torch.float32)
        return acc, esm, struct, prost

def collate_fn(batch):
    ids, esm_list, struct_list, prost_list = zip(*batch)
    esm_pad = pad_sequence(esm_list, batch_first=True)
    struct = torch.stack(struct_list)
    prost = torch.stack(prost_list)
    return ids, esm_pad, struct, prost


# ------------------------------ 加载模型 ------------------------------
# 这里要根据训练时定义的 d_model 和输入维度调整
dummy_dataset = ProteinDataset(list(esm_dict.keys()), esm_dict, struct_dict, prost_dict)
feature_extractor = CrossModalFusionWithMoE(
    esm2_dim=1280,
    foldseek_dim=1024,  # struct dim
    prost5_dim=512,
    d_model=512,
    num_heads=8,
    num_experts=6,
    top_k_experts=2,
    num_fusion_layers=2,
    dropout=0.2
)
model = ProteinRegressionModel(feature_extractor, d_model=512)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)



model.eval()
print("模型加载完成")

# ------------------------------ 推理 ------------------------------
def inference(ids_list, batch_size=BATCH_SIZE):
    dataset = ProteinDataset(ids_list, esm_dict, struct_dict, prost_dict)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    results = []

    with torch.no_grad():
        for ids, esm, struct, prost in loader:
            esm = esm.to(DEVICE)
            struct = struct.to(DEVICE)
            prost = prost.to(DEVICE)

            pred_raw, _ = model(esm, struct, prost)
            pred = torch.sigmoid(pred_raw)  # Sigmoid 输出 [0,1]
            pred_ph = pred.cpu().numpy() * (Y_MAX - Y_MIN) + Y_MIN

            for acc, val in zip(ids, pred_ph):
                results.append((acc, float(val)))

    return results

# ------------------------------ 主程序 ------------------------------
if __name__ == "__main__":
    # 读取待预测 ID 列表
    df_input = pd.read_csv("/data5/heyuhang/mywork/mydata/phtest/test_unique.csv")  # 文件里只包含 ID 列
    ids_list = df_input["Accession"].tolist()

    results = inference(ids_list)

    # 保存预测结果
    df_out = pd.DataFrame(results, columns=["Accession", "Predicted_pH"])
    df_out.to_csv("/data5/heyuhang/mywork/MOE/predictions11.csv", index=False)
    print("预测完成，结果保存到 predictions.csv")
