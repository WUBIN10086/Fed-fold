import json
import time
import gzip
import shutil
import urllib.request
from pathlib import Path

# ============================================================
# 可调参数(大作业建议先用这套"小而干净"的配置,最不容易训崩)
# ============================================================
DATE_START = "2025-07-01"   # 近一年起始(今天约 2026-07)
DATE_END   = "2025-09-30"   # 近一年结束
RES_MAX    = 2.0            # 分辨率上限(Å),越小越干净
LEN_MIN    = 30            # 序列长度下限,过滤太短的肽
LEN_MAX    = 300           # 序列长度上限,省显存、防长序列训崩
MAX_ENTRIES = 1000         # 最多下载多少条;设 None 表示全部下载
SLEEP_SEC   = 0.1          # 每次请求间隔,别把 RCSB 打太狠
RETRY       = 3            # 单个文件下载失败重试次数

out_dir = Path("data/sha_pdb_0703/sha_mmcif_files")
out_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# 检索条件:近一年 + X-ray + 高分辨率 + 只含一个蛋白 entity + 长度范围
#   - polymer_entity_count == 1 且 polymer_entity_count_protein == 1
#     => 整个结构只有一条蛋白链(可能有多份相同拷贝,预处理时取 A 链即可),
#        且不含 DNA/RNA/其他蛋白,正好适合 SoloSeq 单链预测
# ============================================================
def terminal(attribute, operator, value):
    return {
        "type": "terminal",
        "service": "text",
        "parameters": {"attribute": attribute, "operator": operator, "value": value},
    }

payload = {
    "query": {
        "type": "group",
        "logical_operator": "and",
        "nodes": [
            terminal("rcsb_accession_info.initial_release_date", "greater_or_equal", DATE_START),
            terminal("rcsb_accession_info.initial_release_date", "less_or_equal", DATE_END),
            terminal("exptl.method", "exact_match", "X-RAY DIFFRACTION"),
            terminal("rcsb_entry_info.resolution_combined", "less_or_equal", RES_MAX),
            terminal("rcsb_entry_info.polymer_entity_count", "equals", 1),
            terminal("rcsb_entry_info.polymer_entity_count_protein", "equals", 1),
            terminal("entity_poly.rcsb_sample_sequence_length", "greater_or_equal", LEN_MIN),
            terminal("entity_poly.rcsb_sample_sequence_length", "less_or_equal", LEN_MAX),
        ],
    },
    "return_type": "entry",
    "request_options": {
        "return_all_hits": True,
        "results_verbosity": "compact",
        "sort": [
            {"sort_by": "rcsb_accession_info.initial_release_date", "direction": "asc"}
        ],
    },
}

# ---- 发起检索 ----
req = urllib.request.Request(
    "https://search.rcsb.org/rcsbsearch/v2/query",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req) as r:
    resp = json.load(r)

result_set = resp.get("result_set", [])
ids = []
for x in result_set:
    ids.append((x if isinstance(x, str) else x["identifier"]).lower())

print(f"符合条件的结构总数: {len(ids)}")

# ---- 可选:限制下载数量(大作业几百条足够) ----
if MAX_ENTRIES is not None:
    ids = ids[:MAX_ENTRIES]
    print(f"本次实际下载(截断到 MAX_ENTRIES): {len(ids)}")

# ---- 把 ID 列表存下来,后面做去冗余聚类/分 5 份时要用 ----
id_list_path = out_dir.parent / "entry_ids.txt"
id_list_path.write_text("\n".join(ids))
print(f"ID 列表已保存: {id_list_path}")

# ---- 下载循环:断点续传 + 重试 ----
failed = []
skipped = 0

for i, pdb_id in enumerate(ids, 1):
    gz_path = out_dir / f"{pdb_id}.cif.gz"
    cif_path = out_dir / f"{pdb_id}.cif"
    url = f"https://files.rcsb.org/download/{pdb_id}.cif.gz"

    # 已经下过就跳过,支持中断后重跑
    if cif_path.exists() and cif_path.stat().st_size > 0:
        skipped += 1
        if i % 100 == 0 or i == len(ids):
            print(f"{i}/{len(ids)} done")
        continue

    ok = False
    for attempt in range(1, RETRY + 1):
        try:
            urllib.request.urlretrieve(url, gz_path)
            with gzip.open(gz_path, "rb") as fin, open(cif_path, "wb") as fout:
                shutil.copyfileobj(fin, fout)
            gz_path.unlink()
            ok = True
            break
        except Exception as e:
            if gz_path.exists():
                gz_path.unlink()
            if attempt == RETRY:
                failed.append((pdb_id, str(e)))
            else:
                time.sleep(1.0 * attempt)  # 退避后重试

    time.sleep(SLEEP_SEC)

    if i % 100 == 0 or i == len(ids):
        print(f"{i}/{len(ids)} done")

print("\n下载成功:", len(ids) - len(failed) - skipped)
print("已存在跳过:", skipped)
print("失败:", len(failed))
print("输出目录:", out_dir)

if failed:
    print("\n失败列表:")
    for pdb_id, err in failed:
        print(f"{pdb_id}\t{err}")
