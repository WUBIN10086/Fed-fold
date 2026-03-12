#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


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
    parser.add_argument("pdb", nargs="+", type=Path, help="Input PDB file(s)")
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

    args = parser.parse_args(argv)

    for pdb_path in args.pdb:
        if not pdb_path.is_file():
            raise SystemExit(f"File not found: {pdb_path}")

        items = list(
            iter_residue_plddt(
                pdb_path,
                atom_name=args.atom,
                include_icode=args.include_icode,
                dedupe=args.dedupe,
            )
        )
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

