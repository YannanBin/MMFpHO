import pickle
import numpy as np
import torch

# 配置

input_pkl = "/data5/heyuhang/mywork/framework/Demo/tmp_supervised/esm_2bdc5af703d14abf81414d163a23e00b.pkl"   # 原始未池化 ESM2 特征
output_pkl = "/data5/heyuhang/mywork/framework/Demo/tmp_supervised/esm2.pkl"

# 加载 pickle

with open(input_pkl, "rb") as f:
    esm_dict = pickle.load(f)

pooled_dict = {}

for key, arr in esm_dict.items():
# arr: numpy array, shape (L, D)
    pooled = arr.mean(axis=0)  # 平均池化
    pooled_dict[key] = pooled.astype(np.float32)

# 保存新的 pickle

with open(output_pkl, "wb") as f:
    pickle.dump(pooled_dict, f)

# 检查

for k, v in list(pooled_dict.items())[:3]:
    print(f"{k}: shape={v.shape}, dtype={v.dtype}")
