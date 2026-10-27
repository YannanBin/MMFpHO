import os
import numpy as np
import torch
from transformers import T5Tokenizer, T5EncoderModel
import subprocess
import pickle
import pandas as pd   # ← 新增：只为读取 label.csv

# =========================
# 配置参数（你原来的完全保留）
# =========================
pdb_dir = "/data5/heyuhang/mywork/Topt/Seq2strucutre"
foldseek_db_dir = "/data5/heyuhang/mywork/foldseek/Seq2Topt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_file = "/data5/heyuhang/mywork/Topt/data/seqotp_data/structure.pkl"

# =========================
# 新增：读取你的 label.csv，拿到所有合法的 Accession（纯 ID）
# =========================
#LABEL_CSV = "/data5/heyuhang/mywork/mydata/phtest/test_unique.csv"   # ← 改成你的真实路径
LABEL_CSV = "/data5/heyuhang/mywork/Topt/data/sequence_ogt1_dedup.csv"
df = pd.read_csv(LABEL_CSV)
# 假设第一列就是 Accession（如 A0A0A0K2U8）
standard_ids = set(df["Accession"].astype(str).str.strip())

print(f"从 label.csv 加载 {len(standard_ids)} 个标准 Accession")


# =========================
# Step1: Foldseek 提取 3Di （只加了过滤和 ID 统一，其余完全不动）
# =========================
if not os.path.exists(foldseek_db_dir):
    os.makedirs(foldseek_db_dir)

protein_tokens = {}

for pdb_file in os.listdir(pdb_dir):
    if not pdb_file.endswith(".pdb"):
        continue

    # 原始文件名去掉 .pdb
    base_name = os.path.splitext(pdb_file)[0]      # 如: A0A0A0K2U8_best

    # 统一为标准 Accession（去掉 _best）
    if base_name.endswith("_best"):
        protein_id = base_name[:-5]    # → A0A0A0K2U8
    else:
        protein_id = base_name

    # 关键过滤：只处理在 label.csv 中出现的蛋白
    if protein_id not in standard_ids:
        print(f"跳过不在训练集中的蛋白: {protein_id} ({pdb_file})")
        continue

    pdb_path = os.path.join(pdb_dir, pdb_file)
    print(f"\n===== Processing {protein_id} ← {pdb_file} =====")

    db  = os.path.join(foldseek_db_dir, protein_id + "_db")
    aln = os.path.join(foldseek_db_dir, protein_id + "_aln")
    tmp = os.path.join(foldseek_db_dir, protein_id + "_tmp")
    tsv = os.path.join(foldseek_db_dir, protein_id + "_3di.tsv")

    # 1. 创建数据库
    subprocess.run(["foldseek", "createdb", pdb_path, db], check=True)
    
    # 2. 自比对
    subprocess.run(["foldseek", "search", db, db, aln, tmp, "--threads", "32"], check=True)

    # 3. 提取 3Di
    subprocess.run([
        "foldseek", "convertalis",
        db, db, aln, tsv,
        "--format-output", "query,q3di" 
    ], check=True)

    # 4. 解析第一个 3Di 
    token_seq = None
    with open(tsv) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                token_seq = parts[1].lower().strip()
                break 

    if token_seq is None:
        print(f"警告：{protein_id} 未提取到 3Di，跳过")
        # 清理临时文件（下面会执行）
    else:
        protein_tokens[protein_id] = token_seq
        print(f"Success {protein_id} token length = {len(token_seq)}")

    # 5. 清理临时文件（你原来的完美逻辑，完全保留）
    print(f"Cleaning up Foldseek files for {protein_id}...")
    file_prefix = os.path.join(foldseek_db_dir, protein_id)
    try:
        subprocess.run(f"rm -f {file_prefix}*", shell=True, check=True, cwd=foldseek_db_dir)
        subprocess.run(f"rm -rf {file_prefix}_tmp", shell=True, check=False)
        print(f"Cleanup complete for {protein_id}.")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Cleanup failed for {protein_id}. Error: {e}")

print(f"\nFoldseek 阶段完成！成功提取 {len(protein_tokens)} / {len(standard_ids)} 个蛋白的 3Di")

# =========================
# Step2: ProstT5 embedding （完全不动）
# =========================
print("\n开始 ProstT5 编码...")
tokenizer = T5Tokenizer.from_pretrained("/data5/heyuhang/mywork/ProstT5", do_lower_case=False)
model = T5EncoderModel.from_pretrained("/data5/heyuhang/mywork/ProstT5").to(device)
model.eval()
if device.type == "cuda":
    model = model.half()

protein_features = {}

for protein_id, seq in protein_tokens.items():
    input_seq = "<fold2AA> " + " ".join(list(seq))
    inputs = tokenizer(input_seq, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        hidden = outputs.last_hidden_state

    emb = hidden[0, 2:, :].cpu().numpy()   # 去掉 <fold2AA> 和 <pad>
    protein_features[protein_id] = emb

    print(f"Success {protein_id} embedding = {emb.shape}")

# =========================
# Step3: 保存（key 100% 是纯 Accession）
# =========================
with open(output_file, "wb") as f:
    pickle.dump(protein_features, f)

print(f"\nFeatures saved to {output_file}")
print(f"最终包含 {len(protein_features)} 个蛋白，与训练集完美对齐！")