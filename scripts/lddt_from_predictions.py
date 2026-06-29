#!/usr/bin/env python3
"""Compute global lDDT-Cα of predicted structures vs their experimental mmCIF.

Unlike pLDDT (the model's self-confidence), this is a true accuracy metric: it
compares each predicted *unrelaxed.pdb against the deposited structure using
OpenFold's own superposition-free lddt_ca.

Residue alignment: SoloSeq predicts the full input sequence (= the chain's
seqres), and mmcif_parsing.get_atom_coords returns GT aligned to seqres, so the
two arrays are position-aligned and the same length. Mismatches are skipped and
reported (no silent truncation).

Usage:
  python scripts/lddt_from_predictions.py <predictions_dir> <mmcif_dir> -o out.csv
"""
from __future__ import annotations
import argparse, glob, os, sys, csv
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from openfold.np import protein, residue_constants
from openfold.data import mmcif_parsing
from openfold.utils.loss import lddt_ca


def tag_of(fname: str) -> str:
    return os.path.basename(fname).split("_seq_model")[0]


def gt_coords(mmcif_dir, pdb_id, chain_id):
    cif = os.path.join(mmcif_dir, f"{pdb_id}.cif")
    if not os.path.isfile(cif):
        return None, f"no cif {pdb_id}"
    with open(cif) as f:
        parsed = mmcif_parsing.parse(file_id=pdb_id, mmcif_string=f.read())
    if parsed.mmcif_object is None:
        return None, f"unparsable cif {pdb_id}"
    try:
        pos, mask = mmcif_parsing.get_atom_coords(parsed.mmcif_object, chain_id)
    except Exception as e:                       # chain absent / ambiguous
        return None, f"chain {chain_id}: {type(e).__name__}"
    return (pos, mask), None


def compute_lddt(pred_dir, mmcif_dir):
    results, skips = {}, {}
    for pdb in sorted(glob.glob(os.path.join(pred_dir, "*unrelaxed.pdb"))):
        tag = tag_of(pdb)
        pdb_id, chain_id = tag.rsplit("_", 1)
        pdb_id = pdb_id.lower()
        with open(pdb) as f:
            pred_str = f.read()
        try:
            pred = protein.from_pdb_string(pred_str, chain_id=None)
        except Exception as e:
            skips[tag] = f"pred parse {type(e).__name__}"
            continue
        if pred.atom_positions.shape[0] == 0:
            skips[tag] = "empty prediction"
            continue
        gt, err = gt_coords(mmcif_dir, pdb_id, chain_id)
        if gt is None:
            skips[tag] = err
            continue
        gt_pos, gt_mask = gt
        Lp, Lg = pred.atom_positions.shape[0], gt_pos.shape[0]
        if Lp != Lg:
            skips[tag] = f"len mismatch pred={Lp} gt={Lg}"
            continue
        pred_pos = torch.tensor(pred.atom_positions, dtype=torch.float32)
        gtp = torch.tensor(gt_pos, dtype=torch.float32)
        gtm = torch.tensor(gt_mask, dtype=torch.float32)
        score = lddt_ca(pred_pos, gtp, gtm, per_residue=False).item() * 100.0
        results[tag] = score
    return results, skips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_dir")
    ap.add_argument("mmcif_dir")
    ap.add_argument("-o", "--out_csv", required=True)
    args = ap.parse_args()

    res, skips = compute_lddt(args.pred_dir, args.mmcif_dir)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chain", "lddt_ca"])
        for t in sorted(res):
            w.writerow([t, f"{res[t]:.2f}"])

    vals = list(res.values())
    print(f"scored {len(vals)} chains, skipped {len(skips)}")
    if vals:
        v = sorted(vals)
        n = len(v)
        print(f"mean lDDT-Ca {sum(v)/n:.2f} | median {v[n//2]:.2f} "
              f"| min {v[0]:.2f} | max {v[-1]:.2f}")
    if skips:
        from collections import Counter
        reasons = Counter(s.split(" ")[0] + " " + s.split(" ")[1]
                          if len(s.split(" ")) > 1 else s for s in skips.values())
        print("skip reasons:", dict(reasons))
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
