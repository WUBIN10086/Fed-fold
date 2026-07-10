"""
把 MMseqs2 聚类结果（fasta_to_clusterfile.py 的输出）以 cluster 为单位
均匀划分给 N 个联邦学习 client。

保证：同一个 cluster 内的所有链一定落在同一个 client（不跨 client），
从而避免 client 之间存在高相似度序列造成的数据泄漏。

划分策略：按 cluster 大小（链数）降序，贪心地把每个 cluster 分配给
当前累计链数最少的 client（经典的 LPT / greedy number partitioning），
使各 client 的样本量尽量均衡。

对每个 client 生成：
  <output_dir>/client_<i>/mmcif_files/   —— 该 client 的 .cif（拷贝或软链接）
  <output_dir>/client_<i>/pdb_ids.txt    —— 该 client 的 PDB ID 列表
以及一个总的 <output_dir>/split_manifest.json 记录划分情况。
"""
import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path


def parse_clusters(cluster_file: Path):
    """读取 cluster 文件，每行是空格分隔的 {PDBID}_{CHAIN} 列表。"""
    clusters = []
    with open(cluster_file, "r") as f:
        for line in f:
            chains = line.split()
            if chains:
                clusters.append(chains)
    return clusters


def chain_to_pdb_id(chain_name: str) -> str:
    """'1ABC_A' -> '1abc'（PDB ID 不含下划线，取最后一个下划线之前的部分）。"""
    return chain_name.rsplit("_", 1)[0].lower()


def greedy_balance(clusters, num_clients):
    """按链数降序贪心分配，返回 client_idx -> [cluster, ...]。"""
    # 大 cluster 先分，均衡效果更好
    order = sorted(range(len(clusters)), key=lambda i: len(clusters[i]), reverse=True)
    client_chains = [0] * num_clients                 # 每个 client 当前累计链数
    assignment = defaultdict(list)                    # client_idx -> [cluster, ...]
    for ci in order:
        target = min(range(num_clients), key=lambda k: client_chains[k])
        assignment[target].append(clusters[ci])
        client_chains[target] += len(clusters[ci])
    return assignment


def main(args):
    mmcif_dir = Path(args.mmcif_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clusters = parse_clusters(Path(args.cluster_file))
    print(f"读入 cluster 数: {len(clusters)}，总链数: {sum(len(c) for c in clusters)}")

    assignment = greedy_balance(clusters, args.num_clients)

    manifest = {}
    pdb_to_client = {}  # 记录 pdb -> client，检测（理论上不该发生的）跨 client 冲突

    for client_idx in range(args.num_clients):
        client_dir = output_dir / f"client_{client_idx}"
        cif_out_dir = client_dir / "mmcif_files"
        cif_out_dir.mkdir(parents=True, exist_ok=True)

        pdb_ids = []       # 该 client 的去重 PDB 列表（保持出现顺序）
        seen = set()
        chain_count = 0
        for cluster in assignment[client_idx]:
            chain_count += len(cluster)
            for chain in cluster:
                pdb = chain_to_pdb_id(chain)
                if pdb in pdb_to_client and pdb_to_client[pdb] != client_idx:
                    print(
                        f"[警告] PDB {pdb} 的链分散在多个 client，"
                        f"保留在 client_{pdb_to_client[pdb]}，跳过 client_{client_idx}"
                    )
                    continue
                if pdb not in seen:
                    seen.add(pdb)
                    pdb_ids.append(pdb)
                    pdb_to_client[pdb] = client_idx

        # 拷贝 / 软链接对应的 .cif
        copied, missing = 0, []
        for pdb in pdb_ids:
            src = mmcif_dir / f"{pdb}.cif"
            if not src.exists():
                missing.append(pdb)
                continue
            dst = cif_out_dir / f"{pdb}.cif"
            if dst.exists():
                continue
            if args.link:
                os.symlink(os.path.abspath(src), dst)
            else:
                shutil.copy2(src, dst)
            copied += 1

        (client_dir / "pdb_ids.txt").write_text("\n".join(pdb_ids) + "\n")

        manifest[f"client_{client_idx}"] = {
            "num_clusters": len(assignment[client_idx]),
            "num_chains": chain_count,
            "num_pdbs": len(pdb_ids),
            "cif_copied": copied,
            "cif_missing": missing,
        }
        print(
            f"client_{client_idx}: clusters={len(assignment[client_idx])} "
            f"chains={chain_count} pdbs={len(pdb_ids)} cif={copied} "
            f"missing={len(missing)}"
        )

    (output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n划分完成，明细见: {output_dir / 'split_manifest.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cluster_file", type=str, help="fasta_to_clusterfile.py 输出的 cluster 文件")
    parser.add_argument("mmcif_dir", type=str, help="下载好的 .cif 所在目录")
    parser.add_argument("output_dir", type=str, help="划分结果输出目录")
    parser.add_argument("--num_clients", type=int, default=5, help="client 数量")
    parser.add_argument(
        "--link", action="store_true",
        help="用软链接代替拷贝（省磁盘；Windows 上可能需要管理员权限）",
    )
    args = parser.parse_args()
    main(args)
