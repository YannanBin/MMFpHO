import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

# =========================
# 1. 读取 CSV
# =========================
#csv_file = '/data5/heyuhang/mywork/mydata/phtest/test_unique.csv'
csv_file = '/data5/heyuhang/mywork/anlifenxi/enzyme.csv'
df = pd.read_csv(csv_file)

# =========================
# 2. 提取 J-BH 特征列
# =========================
feature_columns = df.columns[4:44]  # J-BH
X = df[feature_columns].values

# =========================
# 3. 归一化到 [0,1]
# =========================
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)

# =========================
# 4. 线性投影到 512 维
# =========================
input_dim = X_norm.shape[1]
#input_dim = X.shape[1]

output_dim = 512

W = np.random.randn(input_dim, output_dim).astype(np.float32)
b = np.random.randn(output_dim).astype(np.float32)

X_proj = np.dot(X_norm, W) + b   # (N, 512)
#X_proj = np.dot(X, W) + b   # (N, 512)

# =========================
# 5. 构建字典
# =========================
accessions = df['Accession'].values
feature_dict = {acc: vec.astype(np.float32) for acc, vec in zip(accessions, X_proj)}

# =========================
# 6. 保存为 pickle（训练框架能直接读取）
# =========================
#output_file = '/data5/heyuhang/mywork/mydata/phtest/features_512d_liner1.pkl'
output_file = '/data5/heyuhang/mywork/anlifenxi/ph_feature/features_512d_liner1.pkl'
with open(output_file, "wb") as f:
    pickle.dump(feature_dict, f)

print(f"🎉 已保存 512 维特征：{len(feature_dict)} 条 -> {output_file}")
