# infer_ablation.py
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
from feature9_ablation import AblationESM2SingleGlobal
from model import ProteinRegressionModel

# ------------------------------ 配置 ------------------------------
# 选择使用哪个全局特征：'foldseek' 或 'prost5'
GLOBAL_FEATURE = 'prost5'  # 改为 'prost5' 可切换到 ProstT5

# 根据选择的特征设置模型路径
if GLOBAL_FEATURE == 'foldseek':
    MODEL_PATH = "/data5/heyuhang/mywork/MOE/best_model_ablation_esm2_foldseek.pt"
    GLOBAL_DIM = 1024
elif GLOBAL_FEATURE == 'prost5':
    MODEL_PATH = "/data5/heyuhang/mywork/MOE/best_model_ablation_esm2_prost5.pt"
    GLOBAL_DIM = 512
else:
    raise ValueError(f"Unknown global feature: {GLOBAL_FEATURE}")

BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Y_MIN, Y_MAX = 0.0, 14.0  # pH 范围

print(f"\n{'='*60}")
print(f"使用消融模型: ESM2 + {GLOBAL_FEATURE.upper()}")
print(f"模型路径: {MODEL_PATH}")
print(f"{'='*60}\n")

# ------------------------------ 加载特征 ------------------------------
def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

esm_dict = load_pkl("/data5/heyuhang/mywork/mydata/phtest/testesm2.pkl")

if GLOBAL_FEATURE == 'foldseek':
    global_dict = load_pkl("/data5/heyuhang/mywork/mydata/phtest/testfinalstructure.pkl")
elif GLOBAL_FEATURE == 'prost5':
    global_dict = load_pkl("/data5/heyuhang/mywork/mydata/phtest/features_512d_liner1.pkl")

print("ESM2:", len(esm_dict))
print(f"{GLOBAL_FEATURE.upper()}:", len(global_dict))

# ------------------------------ Dataset ------------------------------
class ProteinDatasetAblation(Dataset):
    def __init__(self, ids, esm_dict, global_dict, global_feature):
        self.samples = [acc for acc in ids if acc in esm_dict and acc in global_dict]
        self.esm_dict = esm_dict
        self.global_dict = global_dict
        self.global_feature = global_feature
        print(f"有效样本数: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        acc = self.samples[idx]
        esm = torch.tensor(self.esm_dict[acc], dtype=torch.float32)
        
        # 全局特征
        global_feat = self.global_dict[acc]
        if self.global_feature == 'prost5':
            global_feat = np.nan_to_num(global_feat, nan=0.0)
        global_feat = torch.tensor(global_feat, dtype=torch.float32)
        
        return acc, esm, global_feat

def collate_fn(batch):
    ids, esm_list, global_list = zip(*batch)
    esm_pad = pad_sequence(esm_list, batch_first=True)
    global_feat = torch.stack(global_list)
    return ids, esm_pad, global_feat

# ------------------------------ 加载模型 ------------------------------
feature_extractor = AblationESM2SingleGlobal(
    esm2_dim=1280,
    global_dim=GLOBAL_DIM,
    d_model=512,
    num_heads=8,
    num_experts=6,
    top_k_experts=2,
    num_fusion_layers=2,
    dropout=0.2,
    global_feature_name=GLOBAL_FEATURE
)

model = ProteinRegressionModel(feature_extractor, d_model=512)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

print("模型加载完成\n")

# ------------------------------ 推理 ------------------------------
def inference(ids_list, batch_size=BATCH_SIZE):
    dataset = ProteinDatasetAblation(ids_list, esm_dict, global_dict, GLOBAL_FEATURE)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    results = []

    with torch.no_grad():
        for ids, esm, global_feat in loader:
            esm = esm.to(DEVICE)
            global_feat = global_feat.to(DEVICE)

            # 推理（传入 prost5_feat=None）
            pred_raw, _ = model(esm, global_feat, prost5_feat=None)
            pred = torch.sigmoid(pred_raw)  # Sigmoid 输出 [0,1]
            pred_ph = pred.cpu().numpy() * (Y_MAX - Y_MIN) + Y_MIN

            for acc, val in zip(ids, pred_ph):
                results.append((acc, float(val)))

    return results

# ------------------------------ 主程序 ------------------------------
if __name__ == "__main__":
    # 读取待预测 ID 列表
    df_input = pd.read_csv("/data5/heyuhang/mywork/mydata/phtest/test_unique.csv")
    ids_list = df_input["Accession"].tolist()

    print(f"开始推理，共 {len(ids_list)} 个样本...\n")
    results = inference(ids_list)

    # 保存预测结果
    output_file = f"/data5/heyuhang/mywork/MOE/predictions_ablation_esm2_{GLOBAL_FEATURE}.csv"
    df_out = pd.DataFrame(results, columns=["Accession", "Predicted_pH"])
    df_out.to_csv(output_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"预测完成！")
    print(f"结果保存到: {output_file}")
    print(f"预测样本数: {len(results)}")
    print(f"{'='*60}\n")
    
