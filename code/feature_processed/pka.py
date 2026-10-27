import os
import pandas as pd
import subprocess

# ================================
# 解析 PROPKA 输出文件
# ================================
def parse_propka_output(propka_file):
    pkas = []
    seen = set()  # 防止同一残基被重复计入

    with open(propka_file, "r") as f:
        for line in f:
            if not line.strip():
                continue

            parts = line.split()
            if len(parts) < 4:
                continue

            res_name = parts[0]

            # 只接受可电离残基
            if res_name not in {"ASP", "GLU", "CYS", "TYR", "HIS", "LYS", "ARG"}:
                continue

            # 判断是否是“残基起始行”
            if not parts[1].isdigit():
                continue
            if len(parts[2]) != 1:
                continue

            try:
                pKa = float(parts[3])  # ← 对你这个格式是正确的
            except ValueError:
                continue

            resid = (res_name, parts[1], parts[2])
            if resid in seen:
                continue

            seen.add(resid)
            pkas.append((res_name, pKa))

    return pkas


# ================================
# 主程序：批处理
# ================================
csv_file = "/data5/heyuhang/mywork/anlifenxi/enzyme.csv"
df = pd.read_csv(csv_file)

pdb_root = "/data5/heyuhang/mywork/anlifenxi/pdb"
propka_out_dir = "/data5/heyuhang/mywork/framework/data/anli_testPKA"

acidic = {"ASP", "GLU", "CYS", "TYR"}
basic  = {"HIS", "LYS", "ARG"}

results = []
orig_cwd = os.getcwd()

for idx, row in df.iterrows():
    acc = row["Accession"]
    pdb_file = os.path.join(pdb_root, f"{acc}_best.pdb")

    if not os.path.exists(pdb_file):
        print(f"[警告] 找不到 PDB：{pdb_file}")
        results.append([acc, float("nan"), float("nan"), float("nan")])
        continue

    try:
        os.chdir(propka_out_dir)
        subprocess.run(f"propka3 {pdb_file}", shell=True, check=True)

        propka_file = os.path.join(
            propka_out_dir,
            os.path.basename(pdb_file).replace(".pdb", ".pka")
        )

        if not os.path.exists(propka_file):
            raise FileNotFoundError(f"PROPKA 输出不存在: {propka_file}")

        pkas = parse_propka_output(propka_file)

        # 按类别取 pKa
        acid_vals = [p for r, p in pkas if r in acidic]
        basic_vals = [p for r, p in pkas if r in basic]
        all_vals   = [p for _, p in pkas]

        results.append([
            acc,
            sum(acid_vals) / len(acid_vals) if acid_vals else float("nan"),
            sum(basic_vals) / len(basic_vals) if basic_vals else float("nan"),
            sum(all_vals)   / len(all_vals)   if all_vals else float("nan"),
        ])

    except Exception as e:
        print(f"[错误] {acc} 处理失败: {e}")
        results.append([acc, float("nan"), float("nan"), float("nan")])

os.chdir(orig_cwd)

# ================================
# 输出蛋白平均 pKa
# ================================
out_csv = "/data5/heyuhang/mywork/anlifenxi/ph_charged_residue_pka_summary.csv"
summary_df = pd.DataFrame(
    results,
    columns=[
        "Accession",
        "acidic_pKa_mean",
        "basic_pKa_mean",
        "all_pKa_mean",
    ] 
)
summary_df.to_csv(out_csv, index=False)

print(f"✔ 每个蛋白的平均 pKa 已输出：{out_csv}")
