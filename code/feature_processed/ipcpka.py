import pandas as pd
import subprocess
import os
from tempfile import NamedTemporaryFile

# 配置 IPC 脚本路径和模型
IPC_SCRIPT = "/data5/heyuhang/mywork/ipc-2.0.1/scripts/ipc2_pKa_svr_predictor.py"
MODEL_PATH = "/data5/heyuhang/mywork/ipc-2.0.1/models/IPC2_pKa_75_SVR_9.pickle"

# 输入 CSV，必须包含 sequence 和 accession
input_csv = "/data5/heyuhang/mywork/mydata/matched_data.csv"
df = pd.read_csv(input_csv)

charged_aas = ["D","E","C","Y","H","K","R"]
results = []

for idx, row in df.iterrows():
    seq = row['Sequence']
    accession = row['Accession']

    # 临时 CSV 文件用于 IPC 输入
    with NamedTemporaryFile(mode='w', delete=False, suffix=".csv") as temp_input:
        temp_input.write("pdb,chain,aa,position,sequence\n")
        positions = {aa: [] for aa in charged_aas}
        for pos, aa in enumerate(seq, start=1):
            if aa in charged_aas:
                temp_input.write(f"{accession},A,{aa},{pos},{seq}\n")
                positions[aa].append(pos)
        temp_input_name = temp_input.name

    # 临时 CSV 文件用于 IPC 输出
    with NamedTemporaryFile(mode='w', delete=False, suffix=".csv") as temp_output:
        temp_output_name = temp_output.name

    # 调用 IPC 脚本预测
    cmd = f"python3 {IPC_SCRIPT} {MODEL_PATH} {temp_input_name} {temp_output_name}"
    subprocess.run(cmd, shell=True, check=True)

    # 读取 IPC 输出
    ipc_df = pd.read_csv(temp_output_name)

    # 统计每种带电氨基酸平均 pKa
    avg_pka = {}
    for aa in charged_aas:
        values = ipc_df.loc[ipc_df['aa'] == aa, 'pKa']
        avg_pka[f"{aa}_pKa"] = values.mean() if not values.empty else None
    avg_pka['Accession'] = accession
    results.append(avg_pka)

    # 删除临时文件
    os.unlink(temp_input_name)
    os.unlink(temp_output_name)

# 保存最终 CSV
final_df = pd.DataFrame(results)
final_df.to_csv("predicted_pKa_avg.csv", index=False)
print("Done! Output saved to predicted_pKa_avg.csv")
