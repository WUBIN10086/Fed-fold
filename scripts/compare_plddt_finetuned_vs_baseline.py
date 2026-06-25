#!/usr/bin/env python3
"""Compare per-chain mean pLDDT: finetuned checkpoint vs baseline.

Baseline means are read from the baseline pLDDT log (File:/mean_pLDDT pairs).
Finetuned means are computed from the finetuned prediction PDBs using the same
CA-from-B-factor method as scripts/plddt_from_pdb.py.

Writes a per-chain CSV (chain, baseline, finetuned, delta) and prints a summary.
"""
import os, re, glob, csv, argparse, statistics
from pathlib import Path

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from plddt_from_pdb import iter_residue_plddt  # identical pLDDT extraction

BASE = "data/pdb_recent"


def tag_of(fname):
    return os.path.basename(fname).split("_seq_model")[0]


def baseline_means(log_path):
    means = {}
    cur = None
    with open(log_path) as f:
        for line in f:
            if line.startswith("File:"):
                cur = tag_of(line.split("File:", 1)[1].strip())
            elif line.startswith("mean_pLDDT") and cur is not None:
                means[cur] = float(line.split()[1])
                cur = None
    return means


def finetuned_means(pred_dir):
    means = {}
    for pdb in glob.glob(os.path.join(pred_dir, "*unrelaxed.pdb")):
        items = list(iter_residue_plddt(Path(pdb)))
        if not items:
            continue
        scores = [s for _, s in items]
        means[tag_of(pdb)] = sum(scores) / len(scores)
    return means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_log",
                    default=f"{BASE}/soloseq_inference_outputs/plddt_compute.log")
    ap.add_argument("--ft_pred_dir",
                    default=f"{BASE}/soloseq_inference_finetuned/predictions")
    ap.add_argument("--out_csv",
                    default=f"{BASE}/plddt_finetuned_vs_baseline.csv")
    args = ap.parse_args()

    base = baseline_means(args.baseline_log)
    ft = finetuned_means(args.ft_pred_dir)
    common = sorted(set(base) & set(ft))
    print(f"baseline chains: {len(base)} | finetuned chains: {len(ft)} | "
          f"common: {len(common)}")
    only_ft = sorted(set(ft) - set(base))
    only_base = sorted(set(base) - set(ft))
    if only_ft:
        print(f"finetuned-only (no baseline): {len(only_ft)} e.g. {only_ft[:5]}")
    if only_base:
        print(f"baseline-only (no finetuned): {len(only_base)} e.g. {only_base[:5]}")

    rows = [(c, base[c], ft[c], ft[c] - base[c]) for c in common]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chain", "baseline_plddt", "finetuned_plddt", "delta"])
        for r in rows:
            w.writerow([r[0], f"{r[1]:.2f}", f"{r[2]:.2f}", f"{r[3]:+.2f}"])

    deltas = [r[3] for r in rows]
    bmean = statistics.mean(r[1] for r in rows)
    fmean = statistics.mean(r[2] for r in rows)
    dmean = statistics.mean(deltas)
    improved = sum(1 for d in deltas if d > 0)
    worsened = sum(1 for d in deltas if d < 0)
    n = len(rows)
    deltas_sorted = sorted(deltas)

    def pct(p):
        return deltas_sorted[min(n - 1, int(p / 100 * n))]

    print(f"\n=== Per-chain mean pLDDT: finetuned vs baseline (n={n}) ===")
    print(f"baseline mean : {bmean:.2f}")
    print(f"finetuned mean: {fmean:.2f}")
    print(f"mean delta    : {dmean:+.2f}  (median {statistics.median(deltas):+.2f}, "
          f"std {statistics.pstdev(deltas):.2f})")
    print(f"delta p10/p25/p75/p90: {pct(10):+.2f} / {pct(25):+.2f} / "
          f"{pct(75):+.2f} / {pct(90):+.2f}  (min {deltas_sorted[0]:+.2f}, "
          f"max {deltas_sorted[-1]:+.2f})")
    print(f"improved (delta>0): {improved} ({100*improved/n:.1f}%) | "
          f"worsened: {worsened} ({100*worsened/n:.1f}%) | "
          f"unchanged: {n-improved-worsened}")
    big_up = sum(1 for d in deltas if d >= 5)
    big_dn = sum(1 for d in deltas if d <= -5)
    print(f"|delta|>=5: up {big_up} ({100*big_up/n:.1f}%), "
          f"down {big_dn} ({100*big_dn/n:.1f}%)")
    print(f"\nwrote {args.out_csv}")


if __name__ == "__main__":
    main()
