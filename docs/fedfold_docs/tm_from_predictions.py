#!/usr/bin/env python3
"""Compute global TM-score of predicted structures vs their experimental mmCIF.

Like scripts/lddt_from_predictions.py, but TM-score instead of lDDT-Cα. TM-score
needs superposition, so we reuse OpenFold's superimpose() (SVD/Kabsch, RMSD-
minimizing) — the SAME superposition the repo uses for GDT in validation. Note
this is an RMSD-optimal, not a TM-optimal (TMalign), superposition, so absolute
values are a slight lower bound of a TMalign score; but it is applied identically
to baseline and every checkpoint, so relative comparisons are fair.

TM-score = (1/L) * sum_i 1/(1 + (d_i/d0)^2),  d0 = 1.24*(L-15)^(1/3) - 1.8,
normalized by L = number of valid CA residues in the GT (seqres-aligned).

Residue alignment identical to the lDDT script: SoloSeq predicts the full seqres
and get_atom_coords returns GT aligned to seqres, so the CA arrays are
position-aligned and equal length; mismatches are skipped and reported.

Usage:
  python scripts/tm_from_predictions.py <predictions_dir> <mmcif_dir> -o out.csv
"""
from __future__ import annotations
import argparse, glob, os, sys, csv

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from openfold.np import protein, residue_constants
from openfold.data import mmcif_parsing
from openfold.utils.superimposition import superimpose


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


def tm_score(pred_ca, gt_ca, mask):
    """pred_ca, gt_ca: [N,3] tensors; mask: [N] {0,1}. Returns TM in [0,1]."""
    ref = gt_ca.unsqueeze(0)                  # [1,N,3]
    crd = pred_ca.unsqueeze(0)                # [1,N,3]
    msk = mask.unsqueeze(0)                   # [1,N]
    superimposed, _ = superimpose(ref, crd, msk)   # pred aligned onto GT frame
    superimposed = superimposed[0]
    d = torch.sqrt(torch.sum((superimposed - gt_ca) ** 2, dim=-1) + 1e-8)  # [N]
    L = int(mask.sum().item())
    if L < 1:
        return None
    d0 = 1.24 * ((max(L, 19) - 15) ** (1.0 / 3.0)) - 1.8   # clamp so d0>0 for tiny L
    d0 = max(d0, 0.5)
    per_res = 1.0 / (1.0 + (d / d0) ** 2)
    tm = torch.sum(per_res * mask).item() / L
    return tm


def compute_tm(pred_dir, mmcif_dir):
    ca = residue_constants.atom_order["CA"]
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
        pred_ca = torch.tensor(pred.atom_positions[:, ca, :], dtype=torch.float32)
        gt_ca = torch.tensor(gt_pos[:, ca, :], dtype=torch.float32)
        mask_ca = torch.tensor(gt_mask[:, ca], dtype=torch.float32)
        score = tm_score(pred_ca, gt_ca, mask_ca)
        if score is None:
            skips[tag] = "no valid CA"
            continue
        results[tag] = score
    return results, skips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_dir")
    ap.add_argument("mmcif_dir")
    ap.add_argument("-o", "--out_csv", required=True)
    args = ap.parse_args()

    res, skips = compute_tm(args.pred_dir, args.mmcif_dir)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chain", "tm_score"])
        for t in sorted(res):
            w.writerow([t, f"{res[t]:.4f}"])

    vals = sorted(res.values())
    print(f"scored {len(vals)} chains, skipped {len(skips)}")
    if vals:
        n = len(vals)
        print(f"mean TM {sum(vals)/n:.4f} | median {vals[n//2]:.4f} "
              f"| min {vals[0]:.4f} | max {vals[-1]:.4f}")
    if skips:
        from collections import Counter
        reasons = Counter(s.split(" ")[0] + " " + s.split(" ")[1]
                          if len(s.split(" ")) > 1 else s for s in skips.values())
        print("skip reasons:", dict(reasons))
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
