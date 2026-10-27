import os
import pandas as pd
import numpy as np
import subprocess
import re
# ================================
# 根据残基 pKa 计算对应 pH 的电荷
# ================================
def compute_residue_charge(res_name, pKa, pH):
    acidic = {'ASP', 'GLU', 'CYS', 'TYR'}
    basic  = {'HIS', 'LYS', 'ARG'}

    if res_name in acidic:
        return -1.0 / (1.0 + 10 ** (pH - pKa))
    elif res_name in basic:
        return 1.0 / (1.0 + 10 ** (pKa - pH))
    return 0.0

# ================================
# 解析 PROPKA 输出文件
# ================================
def parse_propka_output(propka_file):
    pkas = {}
    with open(propka_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            res_name = parts[0]
            if res_name not in {"ASP", "GLU", "CYS", "TYR", "HIS", "LYS", "ARG"}:
                continue

            # 链 ID
            chain_id = parts[2]

            # 残基编号，保留字符串以防插入码
            res_num = parts[1]

            # pKa 值，去掉可能的星号 *
            pKa_str = parts[3].replace("*", "")
            try:
                pKa = float(pKa_str)
            except ValueError:
                continue

            pkas[(chain_id, res_num)] = (res_name, pKa)
    return pkas
# ================================
# 计算蛋白在给定 pH 下的整体电荷特征
# ================================
def compute_protein_charge_feature(pkas, pH):
    total, acidic_sum, basic_sum = 0.0, 0.0, 0.0
    for (_, _), (res_name, pKa) in pkas.items():
        charge = compute_residue_charge(res_name, pKa, pH)
        total += charge
        if res_name in {'ASP', 'GLU', 'CYS', 'TYR'}:
            acidic_sum += charge
        elif res_name in {'HIS', 'LYS', 'ARG'}:
            basic_sum += charge
    acid_base_ratio = np.nan if basic_sum == 0 else abs(acidic_sum) / basic_sum
    return total, acidic_sum, basic_sum, acid_base_ratio

# ================================
# CSV 配置
# ================================
csv_file = "/data5/heyuhang/mywork/mydata/matched_data.csv"
df = pd.read_csv(csv_file)

if not {'Accession', 'pHopt'}.issubset(df.columns):
    raise ValueError("CSV 必须包含 Accession 和 pHopt 两列")

pdb_root = "/data5/heyuhang/mywork/framework/data/selected_pdbs"
propka_out_dir = "/data5/heyuhang/mywork/framework/data/pka"


# ================================
# 批处理计算
# ================================
results = []

# 保存当前工作目录
orig_cwd = os.getcwd()

for idx, row in df.iterrows():
    acc = row['Accession']
    pdb_file = os.path.join(pdb_root, f"{acc}_best.pdb")
    if not os.path.exists(pdb_file):
        print(f"[警告] 找不到 PDB 文件：{pdb_file}")
        results.append((np.nan, np.nan, np.nan, np.nan))
        continue

    try:
        # 切换到输出目录运行 PROPKA
        os.chdir(propka_out_dir)
        subprocess.run(f"propka3 {pdb_file}", shell=True, check=True)

        # 输出文件路径
        propka_file = os.path.join(propka_out_dir, os.path.basename(pdb_file).replace(".pdb", ".pka"))
        if not os.path.exists(propka_file):
            raise FileNotFoundError(f"PROPKA 输出文件不存在: {propka_file}")

        # 解析 pKa 并计算电荷特征
        pkas = parse_propka_output(propka_file)
        res = compute_protein_charge_feature(pkas, row['pHopt'])
        results.append(res)

    except Exception as e:
        print(f"[错误] {acc} 处理失败: {e}")
        results.append((np.nan, np.nan, np.nan, np.nan))

# 恢复原工作目录
os.chdir(orig_cwd)

# ================================
# 写回 CSV
# ================================
df[['total_charge', 'acidic_charge', 'basic_charge', 'acid_base_ratio']] = pd.DataFrame(results)
df.to_csv(csv_file, index=False)

print(f"✔ PROPKA 批处理完成，特征已写回 CSV，pKa 文件位于：{propka_out_dir}")
