# inference_foldseek_only.py
# 推理脚本：仅使用 Foldseek 全局特征
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# ------------------------------ 配置 ------------------------------
MODEL_PATH = "/data5/heyuhang/mywork/MOE/best_model_foldseek_only.pt"
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Y_MIN, Y_MAX = 0.0, 14.0  # pH 范围

print("\n" + "=" * 60)
print("使用模型: Foldseek only baseline")
print(f"模型路径: {MODEL_PATH}")
print("=" * 60 + "\n")

# ------------------------------ 加载特征 ------------------------------
def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# 注意：这里需要使用测试集的 Foldseek 特征
# 如果测试集特征文件名不同，请修改路径
foldseek_dict = load_pkl("/data5/heyuhang/mywork/mydata/phtest/testfinalstructure.pkl")
print("Foldseek 特征数:", len(foldseek_dict))

# ------------------------------ Dataset ------------------------------
class FoldseekOnlyDataset(Dataset):
    def __init__(self, ids, foldseek_dict):
        self.samples = [acc for acc in ids if acc in foldseek_dict]
        self.foldseek_dict = foldseek_dict
        
        # 统计缺失的样本
        missing = set(ids) - set(self.samples)
        if missing:
            print(f"⚠️  警告: {len(missing)} 个样本缺少 Foldseek 特征")
        
        print(f"有效样本数: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        acc = self.samples[idx]
        foldseek_feat = torch.tensor(self.foldseek_dict[acc], dtype=torch.float32)
        return acc, foldseek_feat


# ------------------------------ 模型定义（与训练完全一致）--------------
class GlobalFeatureModel(nn.Module):
    """
    全局特征专用回归模型
    无需 pooling，直接从固定维度向量预测
    """
    def __init__(self, input_dim=1024):
        super().__init__()
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


# ------------------------------ 加载模型 ------------------------------
model = GlobalFeatureModel(input_dim=1024).to(DEVICE)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()

print("模型加载完成\n")

# ------------------------------ 推理函数 ------------------------------
def inference(ids_list, batch_size=BATCH_SIZE):
    dataset = FoldseekOnlyDataset(ids_list, foldseek_dict)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    results = []

    with torch.no_grad():
        for ids, foldseek_feat in loader:
            foldseek_feat = foldseek_feat.to(DEVICE)

            pred_raw = model(foldseek_feat)
            pred = torch.sigmoid(pred_raw)  # [0,1]
            pred_ph = pred.cpu().numpy() * (Y_MAX - Y_MIN) + Y_MIN

            for acc, val in zip(ids, pred_ph):
                results.append((acc, float(val)))

    return results


# ------------------------------ 主程序 ------------------------------
if __name__ == "__main__":
    df_input = pd.read_csv(
        "/data5/heyuhang/mywork/mydata/phtest/test_unique.csv"
    )
    ids_list = df_input["Accession"].tolist()

    print(f"开始推理，共 {len(ids_list)} 个样本...\n")
    results = inference(ids_list)

    output_file = "/data5/heyuhang/mywork/MOE/predictions_foldseek_only.csv"
    df_out = pd.DataFrame(results, columns=["Accession", "Predicted_pH"])
    df_out.to_csv(output_file, index=False)

    print("\n" + "=" * 60)
    print("预测完成！")
    print(f"结果保存到: {output_file}")
    print(f"预测样本数: {len(results)}")
    if len(results) < len(ids_list):
        print(f"⚠️  缺失样本数: {len(ids_list) - len(results)}")
    print("=" * 60 + "\n")
    
