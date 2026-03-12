import json
import gzip
import shutil
import urllib.request
from pathlib import Path

payload = {
    "query": {
        "type": "group",
        "logical_operator": "and",
        "nodes": [
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_accession_info.initial_release_date",
                    "operator": "greater_or_equal",
                    "value": "2025-01-01"
                }
            },
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_accession_info.initial_release_date",
                    "operator": "less_or_equal",
                    "value": "2026-02-28"
                }
            }
        ]
    },
    "return_type": "entry",
    "request_options": {
        "return_all_hits": True,
        "results_verbosity": "compact",
        "sort": [
            {
                "sort_by": "rcsb_accession_info.initial_release_date",
                "direction": "asc"
            }
        ]
    }
}

# 保存 cif 文件的目录
out_dir = Path("data/pdb_recent/mmcif_files")
out_dir.mkdir(parents=True, exist_ok=True)

req = urllib.request.Request(
    "https://search.rcsb.org/rcsbsearch/v2/query",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

with urllib.request.urlopen(req) as r:
    resp = json.load(r)

result_set = resp.get("result_set", [])
ids = []
for x in result_set:
    if isinstance(x, str):
        ids.append(x.lower())
    else:
        ids.append(x["identifier"].lower())

print(f"found {len(ids)} entries")

failed = []

for i, pdb_id in enumerate(ids, 1):
    gz_path = out_dir / f"{pdb_id}.cif.gz"
    cif_path = out_dir / f"{pdb_id}.cif"
    url = f"https://files.rcsb.org/download/{pdb_id}.cif.gz"

    try:
        urllib.request.urlretrieve(url, gz_path)
        with gzip.open(gz_path, "rb") as fin, open(cif_path, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        gz_path.unlink()
    except Exception as e:
        failed.append((pdb_id, str(e)))
        if gz_path.exists():
            gz_path.unlink()

    if i % 100 == 0 or i == len(ids):
        print(f"{i}/{len(ids)} done")

print("downloaded:", len(ids) - len(failed))
print("failed:", len(failed))
print("output_dir:", out_dir)

if failed:
    print("\nfailed entries:")
    for pdb_id, err in failed:
        print(f"{pdb_id}\t{err}")
