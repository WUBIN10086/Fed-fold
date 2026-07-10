#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

NAME_SUFFIX_TO_STRIP = "_seq_model_esm1b_ptm_unrelaxed"


@dataclass(frozen=True)
class ResidueKey:
    chain_id: str
    resseq: int
    icode: str = ""


def _parse_pdb_atom_line(line: str) -> Optional[Tuple[str, str, int, str, float]]:
    """
    Parse a PDB ATOM/HETATM line using fixed-width columns.

    Returns:
      (atom_name, chain_id, resseq, icode, b_factor) or None if unparsable.
    """
    if len(line) < 66:
        return None
    record = line[0:6]
    if record not in ("ATOM  ", "HETATM"):
        return None

    atom_name = line[12:16].strip()
    chain_id = line[21].strip() or " "
    resseq_str = line[22:26].strip()
    icode = line[26].strip()
    b_factor_str = line[60:66].strip()

    try:
        resseq = int(resseq_str)
        b_factor = float(b_factor_str)
    except ValueError:
        return None

    return atom_name, chain_id, resseq, icode, b_factor


def iter_residue_plddt(
    pdb_path: Path,
    atom_name: str = "CA",
    include_icode: bool = False,
    dedupe: str = "first",
) -> Iterable[Tuple[ResidueKey, float]]:
    """
    Yield (ResidueKey, pLDDT) for one atom per residue.

    pLDDT is taken from the B-factor column.
    """
    seen: Dict[ResidueKey, float] = {}

    with pdb_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parsed = _parse_pdb_atom_line(line)
            if parsed is None:
                continue
            a_name, chain_id, resseq, icode, b_factor = parsed
            if a_name != atom_name:
                continue

            key = ResidueKey(chain_id=chain_id, resseq=resseq, icode=(icode if include_icode else ""))

            if dedupe == "first":
                if key in seen:
                    continue
                seen[key] = b_factor
            elif dedupe == "max":
                prev = seen.get(key)
                if prev is None or b_factor > prev:
                    seen[key] = b_factor
            else:
                raise ValueError(f"Unknown dedupe mode: {dedupe}")

    # Stable iteration: sort by chain then residue number then insertion code.
    for key in sorted(seen.keys(), key=lambda k: (k.chain_id, k.resseq, k.icode)):
        yield key, seen[key]


@dataclass
class PlddtSummary:
    n: int
    mean: float
    min: float
    max: float
    bins: Dict[str, int]
    per_chain: Dict[str, Tuple[int, float, float, float]]  # chain -> (n, mean, min, max)


def summarize_plddt(items: List[Tuple[ResidueKey, float]]) -> PlddtSummary:
    if not items:
        raise ValueError("No residues found (check atom selection and PDB formatting).")

    scores = [s for _, s in items]
    n = len(scores)
    mean = sum(scores) / n
    mn = min(scores)
    mx = max(scores)

    def _bin(score: float) -> str:
        if score < 50:
            return "<50"
        if score < 70:
            return "50-70"
        if score < 90:
            return "70-90"
        return ">=90"

    bins: Dict[str, int] = {"<50": 0, "50-70": 0, "70-90": 0, ">=90": 0}
    for s in scores:
        bins[_bin(s)] += 1

    by_chain: Dict[str, List[float]] = {}
    for key, s in items:
        by_chain.setdefault(key.chain_id, []).append(s)

    per_chain: Dict[str, Tuple[int, float, float, float]] = {}
    for ch, ss in by_chain.items():
        per_chain[ch] = (len(ss), sum(ss) / len(ss), min(ss), max(ss))

    return PlddtSummary(n=n, mean=mean, min=mn, max=mx, bins=bins, per_chain=per_chain)


def _format_pct(k: int, n: int) -> str:
    return f"{(100.0 * k / n):.1f}%"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize pLDDT from a PDB where pLDDT is stored in the B-factor column.",
    )
    parser.add_argument("pdb", nargs="+", type=Path, help="Input PDB file(s) and/or directories")
    parser.add_argument("-a", "--atom", default="CA", help="Atom name to use per residue (default: CA)")
    parser.add_argument("--include-icode", action="store_true", help="Include insertion code in residue key")
    parser.add_argument(
        "--dedupe",
        choices=("first", "max"),
        default="first",
        help="How to handle duplicate residues (altloc etc.): keep first or max pLDDT (default: first)",
    )
    parser.add_argument("-b", "--bottom", type=int, default=10, help="Show lowest N residues (default: 10)")
    parser.add_argument("-t", "--top", type=int, default=10, help="Show highest N residues (default: 10)")
    parser.add_argument("--no-extremes", action="store_true", help="Do not print lowest/highest tables")
    parser.add_argument(
        "--select-threshold",
        type=float,
        default=80.0,
        help="Select proteins with mean pLDDT below this threshold (default: 80.0)",
    )
    parser.add_argument(
        "--selected-csv",
        type=Path,
        default=Path("docs/selectedPDB/SHA.csv"),
        help="CSV output path for selected proteins (default: docs/selectedPDB/SHA.csv)",
    )
    parser.add_argument(
        "--bad-log",
        type=Path,
        default=Path("docs/selectedPDB/SHA_bad_pdb.txt"),
        help="Log path for skipped empty/bad-format files (default: docs/selectedPDB/SHA_bad_pdb.txt)",
    )

    args = parser.parse_args(argv)

    pdb_files: List[Path] = []
    for inp in args.pdb:
        if inp.is_file():
            pdb_files.append(inp)
        elif inp.is_dir():
            pdb_files.extend(sorted(inp.rglob("*.pdb")))
        else:
            raise SystemExit(f"File/Directory not found: {inp}")

    if not pdb_files:
        raise SystemExit("No PDB files found.")

    selected: List[Tuple[str, float]] = []
    bad_files: List[Tuple[Path, str]] = []

    for pdb_path in pdb_files:
        try:
            items = list(
                iter_residue_plddt(
                    pdb_path,
                    atom_name=args.atom,
                    include_icode=args.include_icode,
                    dedupe=args.dedupe,
                )
            )
        except Exception as e:
            bad_files.append((pdb_path, f"read/parse error: {e}"))
            print(f"[SKIP] {pdb_path} -> read/parse error: {e}")
            continue

        if not items:
            bad_files.append((pdb_path, "no residues found (atom selection or PDB formatting issue)"))
            print(f"[SKIP] {pdb_path} -> no residues found")
            continue

        summary = summarize_plddt(items)

        print(f"File: {pdb_path}")
        print(f"== Global (per-residue using {args.atom}) ==")
        print(f"residues\t{summary.n}")
        print(f"mean_pLDDT\t{summary.mean:.2f}")
        print(f"min\t{summary.min:.2f}")
        print(f"max\t{summary.max:.2f}")
        print()

        print("== Bins ==")
        for k in ("<50", "50-70", "70-90", ">=90"):
            v = summary.bins[k]
            print(f"{k}\t{v} ({_format_pct(v, summary.n)})")
        print()

        print("== Per-chain ==")
        for ch in sorted(summary.per_chain.keys()):
            n, mean, mn, mx = summary.per_chain[ch]
            print(f"chain {ch}\tresidues {n}\tmean {mean:.2f}\tmin {mn:.2f}\tmax {mx:.2f}")

        if not args.no_extremes:
            def key_for_sort(x: Tuple[ResidueKey, float]) -> Tuple[float, str, int, str]:
                k, s = x
                return (s, k.chain_id, k.resseq, k.icode)

            lowest = sorted(items, key=key_for_sort)[: max(0, args.bottom)]
            highest = sorted(items, key=key_for_sort, reverse=True)[: max(0, args.top)]

            def _fmt_res(k: ResidueKey) -> str:
                if args.include_icode and k.icode:
                    return f"{k.resseq}{k.icode}"
                return str(k.resseq)

            print()
            print(f"Lowest {args.bottom} residues (chain<TAB>res<TAB>pLDDT):")
            for k, s in lowest:
                print(f"{k.chain_id}\t{_fmt_res(k)}\t{s:.2f}")

            print()
            print(f"Highest {args.top} residues (chain<TAB>res<TAB>pLDDT):")
            for k, s in highest:
                print(f"{k.chain_id}\t{_fmt_res(k)}\t{s:.2f}")

        print()

        mean_score = summary.mean
        if mean_score < args.select_threshold:
            protein_name = pdb_path.stem
            if protein_name.endswith(NAME_SUFFIX_TO_STRIP):
                protein_name = protein_name[: -len(NAME_SUFFIX_TO_STRIP)]
            selected.append((protein_name, mean_score))

    args.selected_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.selected_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["protein_name", "mean_plddt"])
        for protein_name, mean_score in sorted(selected, key=lambda x: x[1]):
            writer.writerow([protein_name, f"{mean_score:.2f}"])

    args.bad_log.parent.mkdir(parents=True, exist_ok=True)
    with args.bad_log.open("w", encoding="utf-8") as f:
        f.write("pdb_path\treason\n")
        for pdb_path, reason in bad_files:
            f.write(f"{pdb_path}\t{reason}\n")

    print(
        f"Wrote {len(selected)} proteins with mean pLDDT < {args.select_threshold:.2f} "
        f"to {args.selected_csv}"
    )
    print(f"Wrote {len(bad_files)} skipped files to {args.bad_log}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

