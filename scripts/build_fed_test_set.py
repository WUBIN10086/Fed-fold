"""
为联邦评测搭建"测试集"：把每个 client 的候选样本（默认=低 pLDDT 微调子集的
kept_labels）按 **整个 cluster** 切成 train / test（同一 cluster 不跨 train/test，
杜绝泄漏），并把所有 client 的 test 组装成一个"并集测试集"，供后续对每个模型
只预测一次即可同时得到「各自 client 测试集」和「全部 client 测试集」两种口径。

对每个 client 产出（写到 <out_dir>/client_i/）：
  train_labels.txt   —— 该 client 微调应使用的训练白名单（传给 --train_filter_path）
  test_labels.txt    —— 该 client 的测试集标签

对并集产出（写到 <out_dir>/all/）：
  test_labels.txt          —— 所有 client 测试标签
  label_client_map.csv     —— label,client（供 eval_tm_matrix.py 归组）
  solo_fasta_dir/          —— 并集测试序列 fasta（从各 client 的全量 fasta 复制）
  solo_alignment_dir/      —— 并集测试 ESM embedding（从各 client 复制/软链）
  mmcif_files/             —— 并集测试对应的 cif（去重复制/软链，用于抽 native）

用法：
  python scripts/build_fed_test_set.py \
      --fed-split-dir data/all_pdb_1y/fed_split \
      --num-clients 5 \
      --cluster-file data/all_pdb_1y/clusters_30.txt \
      --out-dir data/all_pdb_1y/fed_test \
      --test-frac 0.2 --seed 42
"""
import argparse
import csv
import os
import random
import shutil
from pathlib import Path


def read_labels(path: Path) -> list[str]:
    return [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def build_label_to_cluster(cluster_file: Path) -> dict[str, int]:
    """label(大写) -> cluster_id。cluster 文件每行是空格分隔的 PDBID_CHAIN。"""
    mapping: dict[str, int] = {}
    with cluster_file.open("r", encoding="utf-8") as f:
        for cid, line in enumerate(f):
            for tok in line.split():
                mapping[tok.strip().upper()] = cid
    return mapping


def split_by_cluster(labels, label2cluster, test_frac, rng):
    """按整簇划分 train/test。返回 (train_labels, test_labels)，保持输入大小写。"""
    # 把候选 label 按 cluster 分组；找不到 cluster 的各自成簇
    clusters: dict[str, list[str]] = {}
    singleton_seed = 0
    for lab in labels:
        cid = label2cluster.get(lab.upper())
        key = f"c{cid}" if cid is not None else f"s{singleton_seed}"
        if cid is None:
            singleton_seed += 1
        clusters.setdefault(key, []).append(lab)

    total = len(labels)
    target_test = int(round(test_frac * total))

    cluster_keys = list(clusters.keys())
    rng.shuffle(cluster_keys)

    test_labels: list[str] = []
    train_labels: list[str] = []
    for key in cluster_keys:
        members = clusters[key]
        # 还没到测试目标就放进 test，整簇放入
        if len(test_labels) < target_test:
            test_labels.extend(members)
        else:
            train_labels.extend(members)

    return sorted(train_labels), sorted(test_labels)


def copy_or_link(src: Path, dst: Path, link: bool):
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if link:
        os.symlink(os.path.abspath(src), dst)
    elif src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fed-split-dir", type=Path, required=True, help="fed_split 根目录")
    ap.add_argument("--num-clients", type=int, default=5)
    ap.add_argument("--cluster-file", type=Path, required=True, help="MMseqs2 聚类结果文件")
    ap.add_argument("--out-dir", type=Path, required=True, help="测试集输出根目录")
    ap.add_argument("--test-frac", type=float, default=0.2, help="每个 client 划为测试的比例")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--labels-subpath", type=str,
        default="solo_fasta_dir_finetune/_reports/kept_labels.txt",
        help="每个 client 的候选标签清单相对路径（默认=低 pLDDT 微调子集 kept_labels）",
    )
    ap.add_argument("--fasta-subdir", type=str, default="solo_fasta_dir", help="每个 client 的全量 fasta 目录名")
    ap.add_argument("--emb-subdir", type=str, default="solo_alignment_dir", help="每个 client 的 embedding 目录名")
    ap.add_argument("--cif-subdir", type=str, default="mmcif_files", help="每个 client 的 cif 目录名")
    ap.add_argument("--link", action="store_true", help="用软链接代替复制（embedding/cif 省磁盘）")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    label2cluster = build_label_to_cluster(args.cluster_file)

    all_dir = args.out_dir / "all"
    (all_dir / "solo_fasta_dir").mkdir(parents=True, exist_ok=True)
    (all_dir / "solo_alignment_dir").mkdir(parents=True, exist_ok=True)
    (all_dir / "mmcif_files").mkdir(parents=True, exist_ok=True)

    union_test: list[tuple[str, str]] = []  # (label, client_name)

    for i in range(args.num_clients):
        client = f"client_{i}"
        client_dir = args.fed_split_dir / client
        labels_path = client_dir / args.labels_subpath
        if not labels_path.exists():
            print(f"[跳过] {client}: 找不到候选标签 {labels_path}")
            continue

        labels = read_labels(labels_path)
        train_labels, test_labels = split_by_cluster(labels, label2cluster, args.test_frac, rng)

        out_client = args.out_dir / client
        out_client.mkdir(parents=True, exist_ok=True)
        (out_client / "train_labels.txt").write_text("\n".join(train_labels) + "\n", encoding="utf-8")
        (out_client / "test_labels.txt").write_text("\n".join(test_labels) + "\n", encoding="utf-8")
        print(f"{client}: 候选={len(labels)} -> train={len(train_labels)} test={len(test_labels)}")

        # 把该 client 的测试样本汇入并集
        fasta_dir = client_dir / args.fasta_subdir
        emb_dir = client_dir / args.emb_subdir
        cif_dir = client_dir / args.cif_subdir
        missing = []
        for lab in test_labels:
            pdb_id = lab.rsplit("_", 1)[0].lower()
            # fasta
            src_fa = fasta_dir / f"{lab}.fasta"
            if src_fa.exists():
                copy_or_link(src_fa, all_dir / "solo_fasta_dir" / f"{lab}.fasta", link=False)
            else:
                missing.append(f"fasta:{lab}")
            # embedding 子目录
            src_emb = emb_dir / lab
            if src_emb.exists():
                copy_or_link(src_emb, all_dir / "solo_alignment_dir" / lab, link=args.link)
            else:
                missing.append(f"emb:{lab}")
            # cif（按 pdb_id 去重）
            src_cif = cif_dir / f"{pdb_id}.cif"
            if src_cif.exists():
                copy_or_link(src_cif, all_dir / "mmcif_files" / f"{pdb_id}.cif", link=args.link)
            else:
                missing.append(f"cif:{pdb_id}")

            union_test.append((lab, client))

        if missing:
            print(f"    [警告] {client} 有 {len(missing)} 个缺失项（前 5）: {missing[:5]}")

    # 写并集清单与 label->client 映射
    (all_dir / "test_labels.txt").write_text(
        "\n".join(lab for lab, _ in union_test) + "\n", encoding="utf-8"
    )
    with (all_dir / "label_client_map.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "client"])
        w.writerows(union_test)

    print(f"\n并集测试集: {len(union_test)} 条链 -> {all_dir}")
    print(f"  fasta:     {all_dir / 'solo_fasta_dir'}")
    print(f"  embedding: {all_dir / 'solo_alignment_dir'}")
    print(f"  cif:       {all_dir / 'mmcif_files'}")
    print(f"  映射表:    {all_dir / 'label_client_map.csv'}")


if __name__ == "__main__":
    main()
