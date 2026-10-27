import os
import pandas as pd
from Bio.PDB import PDBParser, DSSP

# ----------------------------
# 配置
# ----------------------------
PDB_DIR = "/data5/heyuhang/mywork/anlifenxi/pdb"
CSV_PATH = "/data5/heyuhang/mywork/anlifenxi/enzyme.csv"  # 原始 CSV
RASA_THRESHOLD = 0.25

AA_LIST = ['A','R','N','D','C','Q','E','G','H','I',
           'L','K','M','F','P','S','T','W','Y','V']

MAX_ASA = {
    'A':129, 'R':274, 'N':195, 'D':193, 'C':167, 'Q':223, 'E':223,
    'G':104, 'H':224, 'I':197, 'L':201, 'K':236, 'M':224, 'F':240,
    'P':159, 'S':155, 'T':172, 'W':285, 'Y':263, 'V':174
}

# ----------------------------
# 计算单个 PDB 的 AA 暴露比例
# ----------------------------
def compute_exposed_ratio(pdb_path, threshold=RASA_THRESHOLD):
    parser = PDBParser(QUIET=True)
    pdb_id = os.path.splitext(os.path.basename(pdb_path))[0]
    accession = pdb_id.split('_')[0]

    structure = parser.get_structure(pdb_id, pdb_path)
    model = structure[0]
    dssp = DSSP(model, pdb_path)

    aa_counts = {aa:0 for aa in AA_LIST}
    aa_surface = {aa:0 for aa in AA_LIST}

    for key in dssp.keys():
        aa = dssp[key][1]
        if aa not in AA_LIST:
            continue

        acc_candidate = dssp[key][3]
        maxa = MAX_ASA.get(aa)
        rel_asa = acc_candidate / maxa if acc_candidate > 1 else acc_candidate

        aa_counts[aa] += 1
        if rel_asa >= threshold:
            aa_surface[aa] += 1

    aa_exposed_ratio = {f"{aa}_exposed_ratio": (aa_surface[aa]/aa_counts[aa] if aa_counts[aa]>0 else 0.0) 
                        for aa in AA_LIST}
    aa_exposed_ratio['Accession'] = accession
    return aa_exposed_ratio

# ----------------------------
# 遍历 PDB 文件
# ----------------------------
pdb_files = [os.path.join(PDB_DIR, f) for f in os.listdir(PDB_DIR) if f.endswith(".pdb")]
all_features = []
for pdb in pdb_files:
    try:
        feats = compute_exposed_ratio(pdb)
        all_features.append(feats)
        print(f"Processed {feats['Accession']}")
    except Exception as e:
        print(f"Failed {pdb}: {e}")

features_df = pd.DataFrame(all_features)

# ----------------------------
# 直接在原 CSV 中新增列
# ----------------------------
orig_df = pd.read_csv(CSV_PATH)
# left merge 保留原 CSV 行顺序，如果没有对应 PDB 则 NaN
orig_df = orig_df.merge(features_df, on='Accession', how='left')

# 覆盖原 CSV
orig_df.to_csv(CSV_PATH, index=False)
print(f"Updated original CSV with ASA features: {CSV_PATH}")
