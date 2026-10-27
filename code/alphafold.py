import os
import time
import random
import requests
import pandas as pd
from tqdm import tqdm

# ===== 参数 =====
INPUT_CSV = "/data5/heyuhang/sequence_ogt.csv"
OUTPUT_DIR = "/data5/heyuhang/mywork/Topt/Seq2strucutre"
FAILED_CSV = "/data5/heyuhang/mywork/Topt/download_failed_opt.csv"

# AlphaFold endpoints
AF_API_PRED = "https://alphafold.ebi.ac.uk/api/prediction/{}"

# ===== 随机 User-Agent 列表 =====
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Mozilla/5.0 (X11; Linux x86_64)",
    "Mozilla/5.0 (Windows NT 6.1; Win64; x64)",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:90.0)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_3_0)",
]

def get_headers():
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "application/json, text/plain, */*",
        "Connection": "keep-alive",
    }

# ===== 准备 =====
os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.read_csv(INPUT_CSV)
required_columns = {"Accession", "Sequence", "Xref"}
if not required_columns.issubset(df.columns):
    raise ValueError("CSV 必须包含列：Accession, Sequence, Xref")

failed = []

# ===== 获取 entryIds =====
def fetch_entry_ids_from_api(xref, max_attempts=3):
    """从AlphaFold API获取entryId列表"""
    api_url = AF_API_PRED.format(xref)
    for attempt in range(max_attempts):
        try:
            r = requests.get(api_url, headers=get_headers(), timeout=15)
            if r.status_code == 404:
                return []
            if r.status_code == 429:
                wait = 5 + 5 * attempt
                print(f"  429 from API for {xref}, waiting {wait}s ...")
                time.sleep(wait)
                continue
            if r.status_code >= 500:
                time.sleep(2 + attempt * 3)
                continue
            if r.status_code == 200:
                try:
                    js = r.json()
                    # 修正: AlphaFold API返回的是单个对象或对象数组
                    # 需要根据实际返回格式调整
                    if isinstance(js, list):
                        entry_ids = [item["entryId"] for item in js if "entryId" in item]
                    elif isinstance(js, dict) and "entryId" in js:
                        entry_ids = [js["entryId"]]
                    else:
                        return []
                    return entry_ids
                except (ValueError, KeyError, TypeError):
                    return []
            time.sleep(1 + attempt * 2)
        except Exception as e:
            print(f"  API请求异常: {e}")
            time.sleep(2 + attempt * 3)
    return []

# ===== 主循环 =====
print("\n开始从 AlphaFold 下载结构（先通过 API 确认 entryId）...\n")

for _, row in tqdm(df.iterrows(), total=len(df)):
    acc = row["Accession"]
    seq = row["Sequence"]
    xref = str(row["Xref"]).strip()

    if not xref or xref.lower() == "nan":
        print(f"✘ 跳过 {acc}：Xref 为空")
        failed.append([acc, seq, "Xref empty"])
        continue

    out_path = os.path.join(OUTPUT_DIR, f"{acc}_best.pdb")
    if os.path.exists(out_path):
        size = os.path.getsize(out_path)
        print(f"✔ 已存在 {acc} ({size} bytes)，跳过")
        continue

    # 获取entryId
    entry_ids = fetch_entry_ids_from_api(xref)
    if not entry_ids:
        print(f"✘ {acc}：AlphaFold 上无预测或 API 调用失败")
        failed.append([acc, seq, "No prediction/API error"])
        time.sleep(random.uniform(3, 5))
        continue

    # 尝试第一个 entryId
    entry_id = entry_ids[0]
    # entryId 已经包含 AF- 前缀，直接使用
    file_url = f"https://alphafold.ebi.ac.uk/files/{entry_id}-model_v6.pdb"
    print(f"→ {acc} 使用 entryId {entry_id} 下载...")

    success = False
    for attempt in range(5):
        try:
            r = requests.get(file_url, headers=get_headers(), timeout=25)
            if r.status_code == 404:
                print(f"✘ {acc} HTTP 404 → 跳过")
                failed.append([acc, seq, f"404 Not Found (entry {entry_id})"])
                break
            if r.status_code == 429:
                wait = 10 + 10 * attempt
                print(f"⚠️ 429 Too Many Requests for {acc}, wait {wait}s ...")
                time.sleep(wait)
                continue
            if r.status_code >= 500:
                print(f"✘ {acc} HTTP {r.status_code} (server error), retrying...")
                time.sleep(2 + attempt * 3)
                continue
            if r.status_code == 200:
                content = r.content
                if len(content) < 200:
                    print(f"✘ {acc} 下载到异常小文件 ({len(content)} bytes)，尝试重试 ({attempt+1}/5)")
                    time.sleep(3 + attempt * 2)
                    continue
                with open(out_path, "wb") as fw:
                    fw.write(content)
                print(f"✔ 成功下载 {acc} ({len(content)} bytes)")
                success = True
                break
            print(f"✘ {acc} HTTP {r.status_code}（尝试 {attempt+1}/5）")
            time.sleep(2 + attempt * 2)
        except Exception as e:
            print(f"✘ {acc} 异常：{e}（尝试 {attempt+1}/5）")
            time.sleep(4 + attempt * 3)

    if not success:
        failed.append([acc, seq, f"Download failed (entry {entry_id})"])

    # 请求间隔,避免被限流
    time.sleep(random.uniform(2, 4))

# ===== 保存失败清单 =====
if failed:
    fail_df = pd.DataFrame(failed, columns=["Accession", "Sequence", "Reason"])
    fail_df.to_csv(FAILED_CSV, index=False)
    print(f"\n❗ 有 {len(failed)} 条下载失败，已保存到：{FAILED_CSV}")
else:
    print("\n🎉 全部下载成功！")

print("\n任务完成！")