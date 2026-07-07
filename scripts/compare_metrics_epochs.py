#!/usr/bin/env python3
"""Paired per-chain comparison of pLDDT, lDDT-Ca, and TM-score for each finetuned
epoch checkpoint (ep1/ep2/ep3) vs the same-lineage baseline.

For each metric and epoch, restricts to chains scored in BOTH baseline and that
epoch (paired), and reports mean before/after, mean/median delta, and win/loss
counts. This is robust to differing scored-subset sizes (a plain difference of
two dataset means is not).

Sources:
  pLDDT baseline : soloseq_inference_outputs/plddt_compute.log  (File:/mean_pLDDT)
  pLDDT epoch    : ckpt_eval_epN/plddt_per_chain.csv            (headerless tag,val)
  lDDT  baseline : baseline_lddt.csv                            (chain,lddt_ca)
  lDDT  epoch    : ckpt_eval_epN/lddt_per_chain.csv
  TM    baseline : baseline_tm.csv                              (chain,tm_score)
  TM    epoch    : ckpt_eval_epN/tm_per_chain.csv
"""
import os, csv, statistics as st

BASE = "data/pdb_recent"
EPOCHS = {"ep1": "0-10000", "ep2": "1-20000", "ep3": "2-30000"}


def load_csv(path, headerless=False):
    d = {}
    if not os.path.isfile(path):
        return d
    with open(path) as f:
        rows = list(csv.reader(f))
    start = 0
    if not headerless and rows and not _isnum(rows[0][1]):
        start = 1
    for r in rows[start:]:
        if len(r) >= 2 and _isnum(r[1]):
            d[r[0]] = float(r[1])
    return d


def _isnum(s):
    try:
        float(s); return True
    except (ValueError, IndexError):
        return False


def baseline_plddt(log_path):
    means, cur = {}, None
    with open(log_path) as f:
        for line in f:
            if line.startswith("File:"):
                cur = os.path.basename(line.split("File:", 1)[1].strip()).split("_seq_model")[0]
            elif line.startswith("mean_pLDDT") and cur is not None:
                means[cur] = float(line.split()[1]); cur = None
    return means


def summarize(base, epo):
    common = sorted(set(base) & set(epo))
    n = len(common)
    if n == 0:
        return None
    deltas = [epo[c] - base[c] for c in common]
    bmean = st.mean(base[c] for c in common)
    emean = st.mean(epo[c] for c in common)
    up = sum(1 for d in deltas if d > 0)
    dn = sum(1 for d in deltas if d < 0)
    return dict(n=n, bmean=bmean, emean=emean,
                dmean=st.mean(deltas), dmed=st.median(deltas),
                up=up, dn=dn, up_pct=100 * up / n, dn_pct=100 * dn / n)


def main():
    # baseline sources
    base = {
        "pLDDT": baseline_plddt(f"{BASE}/soloseq_inference_outputs/plddt_compute.log"),
        "lDDT":  load_csv(f"{BASE}/baseline_lddt.csv"),
        "TM":    load_csv(f"{BASE}/baseline_tm.csv"),
    }
    # epoch sources
    epo = {}
    for tag in EPOCHS:
        d = f"{BASE}/ckpt_eval_{tag}"
        epo[tag] = {
            "pLDDT": load_csv(f"{d}/plddt_per_chain.csv", headerless=True),
            "lDDT":  load_csv(f"{d}/lddt_per_chain.csv"),
            "TM":    load_csv(f"{d}/tm_per_chain.csv"),
        }

    for metric in ("pLDDT", "lDDT", "TM"):
        fmt = "{:.4f}" if metric == "TM" else "{:.2f}"
        print(f"\n=== {metric}: finetuned epoch vs baseline (paired per-chain) ===")
        print(f"baseline chains scored: {len(base[metric])}")
        print(f"{'epoch':<20} {'n':>5} {'base':>9} {'ft':>9} {'Δmean':>9} {'Δmed':>9} "
              f"{'%up':>6} {'%down':>7}")
        for tag, step in EPOCHS.items():
            s = summarize(base[metric], epo[tag][metric])
            if s is None:
                print(f"{tag} ({step})   no common chains"); continue
            sign = "+" if s["dmean"] >= 0 else ""
            print(f"{tag+' ('+step+')':<20} {s['n']:>5} "
                  f"{fmt.format(s['bmean']):>9} {fmt.format(s['emean']):>9} "
                  f"{sign+fmt.format(s['dmean']):>9} "
                  f"{('+' if s['dmed']>=0 else '')+fmt.format(s['dmed']):>9} "
                  f"{s['up_pct']:>5.1f} {s['dn_pct']:>6.1f}")

    # combined per-chain CSV (chains common to baseline & all three epochs, per metric)
    out = f"{BASE}/metrics_epochs_vs_baseline.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "chain", "baseline", "ep1", "ep2", "ep3"])
        for metric in ("pLDDT", "lDDT", "TM"):
            common = set(base[metric])
            for tag in EPOCHS:
                common &= set(epo[tag][metric])
            for c in sorted(common):
                w.writerow([metric, c, base[metric][c]] +
                           [epo[t][metric][c] for t in EPOCHS])
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
