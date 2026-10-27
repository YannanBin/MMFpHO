import pandas as pd
import numpy as np
from metrics import calculate_all_metrics, print_metrics

# ========= 路径 =========
pred_file = "/data5/heyuhang/mywork/Topt/data/seqotp_data/Catopt.csv"
label_file = "/data5/heyuhang/mywork/Topt/data/seqotp_data/test_Seq2_filtered.csv"
pred_column = "pred_topt"   # 使用 Ensemble 作为最终预测

# ========= 读取预测结果 =========
pred_df = pd.read_csv(pred_file)

# prediction.csv: id, RLATtr, SVR, Ensemble
pred_df = pred_df.set_index("Accession")

# ========= 读取真实标签 =========
label_df = pd.read_csv(label_file)

# test_clean.csv: Accession, pHopt
label_df = label_df.set_index("Accession")

# ========= 对齐 accession =========
merged = label_df.join(pred_df, how="inner")

print(f"匹配到的评估样本数: {len(merged)}")

# ========= 提取真实值 & 预测值 =========
y_true = merged["Topt"].astype(float).values
y_pred = merged[pred_column].astype(float).values

# ========= 计算指标 =========
metrics = calculate_all_metrics(y_true, y_pred)

# ========= 打印 =========
print_metrics(metrics, title="XGBoost Evaluation (Ensemble)")
