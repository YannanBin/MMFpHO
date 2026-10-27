import os
import pandas as pd
import subprocess

# ================================
# 路径配置
# ================================
csv_file = "/data5/heyuhang/mywork/anlifenxi/enzyme.csv"
pdb_root = "/data5/heyuhang/mywork/anlifenxi/pdb"
propka_out_dir ="/data5/heyuhang/mywork/framework/data/anli_testPKA"

# 确保输出目录存在
os.makedirs(propka_out_dir, exist_ok=True)

# ================================
# 读取 CSV
# ================================
df = pd.read_csv(csv_file)

if 'Accession' not in df.columns:
    raise ValueError("CSV 必须包含 Accession 列")

# ================================
# 批量运行 PROPKA
# ================================
for acc in df['Accession']:
    pdb_file = os.path.join(pdb_root, f"{acc}_best.pdb")

    if not os.path.exists(pdb_file):
        print(f"[跳过] 找不到 PDB: {pdb_file}")
        continue

    # PROPKA 输出文件名
    pka_file = os.path.join(
        propka_out_dir,
        f"{acc}_best.pka"
    )

    # 已存在就跳过（防止重复跑）
    if os.path.exists(pka_file):
        print(f"[已存在] 跳过: {acc}")
        continue

    try:
        subprocess.run(
            ["propka3", pdb_file],
            cwd=propka_out_dir,   # 指定输出目录，不用 chdir
            check=True
        )
        print(f"[完成] PROPKA: {acc}")

    except subprocess.CalledProcessError:
        print(f"[失败] PROPKA 运行错误: {acc}")

print(f"✔ 全部完成，.pka 位于: {propka_out_dir}")
