#!/usr/bin/env python3
"""Carve a held-out validation split for SoloSeq finetuning.

Holds out whole PDB structures (so identical chains in homomers never straddle
the train/val boundary) from the matched pool
    matched = embeddings_output_dir  ∩  solo_chain_data_cache.json  ∩  selected_sequences/*.cif
and materialises it as symlinks (non-destructive: originals are untouched).

Outputs:
  data/pdb_recent/val_cifs/          symlinks to each val structure's .cif
  data/pdb_recent/val_embeddings/    symlinks to each val chain's ESM embedding dir
  data/pdb_recent/train_filter.txt   include-list of TRAIN chains (matched minus val)
"""
import json, os, random, argparse

BASE = "data/pdb_recent"
EMB = f"{BASE}/embeddings_output_dir"
CIF = f"{BASE}/selected_sequences"
CACHE = f"{BASE}/solo_chain_data_cache.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_val_chains", type=int, default=120)
    ap.add_argument("--max_chains_per_struct", type=int, default=8,
                    help="skip structures larger than this when picking val "
                         "(keeps val balanced; giant homomers stay in train)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    cache = set(json.load(open(CACHE)).keys())
    emb = [e for e in os.listdir(EMB) if os.path.isdir(os.path.join(EMB, e))]
    cif_pdbs = {f[:-4] for f in os.listdir(CIF) if f.endswith(".cif")}

    matched = [c for c in emb if c in cache and c.rsplit("_", 1)[0] in cif_pdbs]
    by_pdb = {}
    for c in matched:
        by_pdb.setdefault(c.rsplit("_", 1)[0], []).append(c)
    print(f"matched: {len(matched)} chains across {len(by_pdb)} structures")

    # deterministically choose val structures
    eligible = sorted(p for p, cs in by_pdb.items()
                      if len(cs) <= args.max_chains_per_struct)
    rng = random.Random(args.seed)
    rng.shuffle(eligible)

    val_pdbs, val_chains = [], 0
    for p in eligible:
        if val_chains >= args.target_val_chains:
            break
        val_pdbs.append(p)
        val_chains += len(by_pdb[p])
    val_pdbs = set(val_pdbs)

    val_chain_list = sorted(c for p in val_pdbs for c in by_pdb[p])
    train_chain_list = sorted(c for c in matched
                              if c.rsplit("_", 1)[0] not in val_pdbs)
    print(f"VAL:   {len(val_chain_list)} chains / {len(val_pdbs)} structures")
    print(f"TRAIN: {len(train_chain_list)} chains / "
          f"{len(by_pdb) - len(val_pdbs)} structures")
    assert not (set(val_chain_list) & set(train_chain_list)), "leak!"

    if args.dry_run:
        print("dry run — nothing written")
        return

    val_cif_dir = f"{BASE}/val_cifs"
    val_emb_dir = f"{BASE}/val_embeddings"
    os.makedirs(val_cif_dir, exist_ok=True)
    os.makedirs(val_emb_dir, exist_ok=True)
    abs_ = os.path.abspath

    for p in sorted(val_pdbs):
        dst = os.path.join(val_cif_dir, f"{p}.cif")
        if not os.path.lexists(dst):
            os.symlink(abs_(os.path.join(CIF, f"{p}.cif")), dst)
    for c in val_chain_list:
        dst = os.path.join(val_emb_dir, c)
        if not os.path.lexists(dst):
            os.symlink(abs_(os.path.join(EMB, c)), dst)

    with open(f"{BASE}/train_filter.txt", "w") as f:
        f.write("\n".join(train_chain_list) + "\n")

    print(f"wrote {val_cif_dir}/ ({len(val_pdbs)} cifs), "
          f"{val_emb_dir}/ ({len(val_chain_list)} dirs), "
          f"{BASE}/train_filter.txt ({len(train_chain_list)} chains)")


if __name__ == "__main__":
    main()
