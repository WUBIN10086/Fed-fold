import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path


def load_pdb_ids(csv_path: Path) -> list[str]:
    pdb_ids = []
    seen = set()

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "protein_name" not in (reader.fieldnames or []):
            raise ValueError("CSV must contain a 'protein_name' column.")

        for row in reader:
            name = (row.get("protein_name") or "").strip()
            if not name:
                continue
            pdb_id = name.split("_", 1)[0].strip().lower()
            if pdb_id and pdb_id not in seen:
                seen.add(pdb_id)
                pdb_ids.append(pdb_id)

    return pdb_ids


def is_target_structure_file(file_name_lower: str) -> bool:
    return (
        file_name_lower.endswith(".cif")
        or file_name_lower.endswith(".cif.gz")
        or file_name_lower.endswith(".mmcif")
        or file_name_lower.endswith(".mmcif.gz")
    )


def matches_pdb_id(file_name_lower: str, pdb_id: str) -> bool:
    if not file_name_lower.startswith(pdb_id):
        return False
    if not is_target_structure_file(file_name_lower):
        return False

    rest = file_name_lower[len(pdb_id) :]
    # Avoid false matches like "10ic" matching "10ic0.cif".
    if rest and rest[0].isalnum():
        return False
    return True


def leading_alnum_token(file_name_lower: str) -> str:
    idx = 0
    length = len(file_name_lower)
    while idx < length and file_name_lower[idx].isalnum():
        idx += 1
    return file_name_lower[:idx]


def write_reports(
    report_dir: Path,
    copied_manifest: list[tuple[str, str, str]],
    missing_ids: list[str],
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = report_dir / "copied_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pdb_id", "source_file", "destination_file"])
        writer.writerows(copied_manifest)

    missing_path = report_dir / "missing_ids.txt"
    with missing_path.open("w", encoding="utf-8") as f:
        for pdb_id in missing_ids:
            f.write(f"{pdb_id}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract selected mmCIF files from a source folder."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("docs/selectedPDB/SHA.csv"),
        help="Path to CSV with a 'protein_name' column.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/pdb_recent/sha_mmcif_files"),
        help="Directory containing original mmCIF files.",
    )
    parser.add_argument(
        "--dst-dir",
        type=Path,
        default=Path("data/pdb_recent/selected_sha_mmcif_files"),
        help="Destination directory for copied files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print statistics without copying files.",
    )
    args = parser.parse_args()

    csv_path = args.csv_path
    source_dir = args.source_dir
    dst_dir = args.dst_dir

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    pdb_ids = load_pdb_ids(csv_path)
    source_files = [p for p in source_dir.iterdir() if p.is_file()]
    pdb_id_set = set(pdb_ids)
    matches_by_id: dict[str, list[Path]] = defaultdict(list)

    for p in source_files:
        file_name_lower = p.name.lower()
        if not is_target_structure_file(file_name_lower):
            continue
        token = leading_alnum_token(file_name_lower)
        if token in pdb_id_set and matches_pdb_id(file_name_lower, token):
            matches_by_id[token].append(p)

    files_to_copy = []
    copied_manifest = []
    missing_ids = []
    queued_names = set()

    for pdb_id in pdb_ids:
        matches = matches_by_id.get(pdb_id, [])
        if not matches:
            missing_ids.append(pdb_id)
            continue

        for src in matches:
            src_name_lower = src.name.lower()
            if src_name_lower in queued_names:
                continue
            queued_names.add(src_name_lower)
            dst = dst_dir / src.name
            files_to_copy.append((src, dst))
            copied_manifest.append((pdb_id, str(src), str(dst)))

    if not args.dry_run:
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src, dst in files_to_copy:
            shutil.copy2(src, dst)

    write_reports(dst_dir, copied_manifest, missing_ids)

    total = len(pdb_ids)
    missing = len(missing_ids)
    copied = len(files_to_copy)
    ratio = (missing / total * 100.0) if total else 0.0

    print(f"Total unique pdb_id: {total}")
    print(f"Matched files      : {copied}")
    print(f"Missing pdb_id     : {missing} ({ratio:.2f}%)")
    print(f"Dry run            : {args.dry_run}")
    print(f"Manifest           : {dst_dir / 'copied_manifest.csv'}")
    print(f"Missing list       : {dst_dir / 'missing_ids.txt'}")


if __name__ == "__main__":
    main()
