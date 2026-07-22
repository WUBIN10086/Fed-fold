"""
把多个模型在"并集测试集"上算出的 per-chain TM-score（tmscore_from_pdb.py 的输出）
汇总成一张对比矩阵：

  行 = 模型（如 before / client_0_after / ... / global）
  列 = 测试集（各 client 各自的测试集 + 全部 client 的并集 all）
  格 = 该模型在该测试子集上的平均 TM-score

因为每个模型都是在同一个并集测试集上预测的，这里只需按 label 归属（label_client_map.csv）
切片即可同时得到「各自 client」与「全部」两种口径，无需重复预测。

用法：
  python scripts/eval_tm_matrix.py \
      --label-map data/all_pdb_1y/fed_test/all/label_client_map.csv \
      --scores \
          before=data/all_pdb_1y/fed_test/tm_before.csv \
          client_0_after=data/all_pdb_1y/fed_test/tm_client_0_after.csv \
          client_1_after=data/all_pdb_1y/fed_test/tm_client_1_after.csv \
          global=data/all_pdb_1y/fed_test/tm_global.csv \
      --out-prefix data/all_pdb_1y/fed_test/tm_matrix
"""
import argparse
import csv
import statistics
from pathlib import Path


def read_label_map(path: Path) -> dict[str, str]:
    """label(小写) -> client 名。"""
    m: dict[str, str] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lab = (row.get("label") or "").strip().lower()
            client = (row.get("client") or "").strip()
            if lab and client:
                m[lab] = client
    return m


def read_scores(path: Path) -> dict[str, float]:
    """读取 tmscore_from_pdb.py 输出，返回 label(小写) -> tm_selected。"""
    scores: dict[str, float] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lab = (row.get("label") or "").strip().lower()
            val = (row.get("tm_selected") or "").strip()
            status = (row.get("status") or "ok").strip()
            if not lab or not val or status != "ok":
                continue
            try:
                scores[lab] = float(val)
            except ValueError:
                continue
    return scores


def summarize(vals: list[float]) -> tuple[int, float, float]:
    if not vals:
        return 0, float("nan"), float("nan")
    return len(vals), statistics.fmean(vals), statistics.median(vals)


def parse_kv(pairs: list[str]) -> list[tuple[str, str]]:
    out = []
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"--scores 需要 name=path 形式，收到: {p}")
        name, path = p.split("=", 1)
        out.append((name.strip(), path.strip()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label-map", type=Path, required=True, help="build_fed_test_set.py 产出的 label_client_map.csv")
    ap.add_argument("--scores", nargs="+", required=True, help="name=per_chain_tm.csv，按行（模型）依次给出")
    ap.add_argument("--out-prefix", type=Path, required=True, help="输出前缀，会生成 <prefix>_long.csv 与 <prefix>_mean.csv")
    args = ap.parse_args()

    label_map = read_label_map(args.label_map)
    clients = sorted(set(label_map.values()))
    subsets = clients + ["all"]

    models = parse_kv(args.scores)

    # long 格式：model, testset, count, mean, median
    long_rows = []
    # mean 透视：model -> {subset: mean}
    mean_pivot: dict[str, dict[str, float]] = {}

    for model_name, csv_path in models:
        scores = read_scores(Path(csv_path))
        mean_pivot[model_name] = {}
        for subset in subsets:
            if subset == "all":
                vals = [v for lab, v in scores.items() if lab in label_map]
            else:
                vals = [v for lab, v in scores.items() if label_map.get(lab) == subset]
            n, mean, med = summarize(vals)
            long_rows.append([model_name, subset, n, f"{mean:.4f}", f"{med:.4f}"])
            mean_pivot[model_name][subset] = mean

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    long_path = Path(f"{args.out_prefix}_long.csv")
    mean_path = Path(f"{args.out_prefix}_mean.csv")

    with long_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "testset", "count", "mean_tm", "median_tm"])
        w.writerows(long_rows)

    with mean_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model"] + subsets)
        for model_name, _ in models:
            w.writerow([model_name] + [f"{mean_pivot[model_name][s]:.4f}" for s in subsets])

    # 控制台打印平均值透视表
    col_w = max([len(m) for m, _ in models] + [12])
    header = "model".ljust(col_w) + "".join(s.rjust(12) for s in subsets)
    print(header)
    print("-" * len(header))
    for model_name, _ in models:
        line = model_name.ljust(col_w) + "".join(
            f"{mean_pivot[model_name][s]:.4f}".rjust(12) for s in subsets
        )
        print(line)

    print(f"\n逐项明细: {long_path}")
    print(f"平均值矩阵: {mean_path}")


if __name__ == "__main__":
    main()
