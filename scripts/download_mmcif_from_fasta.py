#!/usr/bin/env python3
"""Download the RCSB mmCIF files referenced by a directory of FASTA files."""

import argparse
import gzip
import shutil
import time
import urllib.request
from pathlib import Path


def pdb_ids_from_fasta_dir(fasta_dir: Path) -> list[str]:
    ids = {
        path.stem.rsplit("_", 1)[0].lower()
        for path in fasta_dir.glob("*.fasta")
        if "_" in path.stem
    }
    if not ids:
        raise ValueError(f"No <pdb_id>_<chain>.fasta files found in {fasta_dir}")
    return sorted(ids)


def download_one(pdb_id: str, output_dir: Path, retries: int) -> str | None:
    output = output_dir / f"{pdb_id}.cif"
    if output.exists() and output.stat().st_size > 0:
        return None

    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif.gz"
    temporary = output.with_suffix(".cif.gz.part")
    error = None
    for attempt in range(1, retries + 1):
        try:
            urllib.request.urlretrieve(url, temporary)
            with gzip.open(temporary, "rb") as source, output.open("wb") as target:
                shutil.copyfileobj(source, target)
            temporary.unlink(missing_ok=True)
            return None
        except Exception as exc:  # network and HTTP failures are reported together
            error = str(exc)
            temporary.unlink(missing_ok=True)
            output.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(attempt)
    return error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fasta_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--delay", type=float, default=0.05)
    args = parser.parse_args()

    if not args.fasta_dir.is_dir():
        raise FileNotFoundError(args.fasta_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pdb_ids = pdb_ids_from_fasta_dir(args.fasta_dir)
    failures = []
    for index, pdb_id in enumerate(pdb_ids, 1):
        error = download_one(pdb_id, args.output_dir, args.retries)
        if error is not None:
            failures.append((pdb_id, error))
        if index % 50 == 0 or index == len(pdb_ids):
            print(f"{index}/{len(pdb_ids)} mmCIF processed")
        time.sleep(args.delay)

    failure_log = args.output_dir / "download_failures.txt"
    failure_log.write_text(
        "".join(f"{pdb_id}\t{error}\n" for pdb_id, error in failures),
        encoding="utf-8",
    )
    if failures:
        raise RuntimeError(
            f"{len(failures)} mmCIF downloads failed; see {failure_log}"
        )
    print(f"mmCIF ready: {args.output_dir}")


if __name__ == "__main__":
    main()
