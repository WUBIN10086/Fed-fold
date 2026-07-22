import argparse
import os
import shutil


def read_skipped_labels(skipped_file):
    labels = set()
    with open(skipped_file, "r") as infile:
        for line in infile:
            label = line.strip()
            if label:
                labels.add(label)
    return labels


def main(args):
    os.makedirs(args.output_fasta_dir, exist_ok=True)
    skipped_labels = read_skipped_labels(args.skipped_file)

    kept_files = 0
    dropped_files = 0
    unsupported_files = 0
    dropped_labels = []

    for fname in os.listdir(args.input_fasta_dir):
        src_path = os.path.join(args.input_fasta_dir, fname)
        if not os.path.isfile(src_path):
            continue

        stem, ext = os.path.splitext(fname)
        if ext.lower() not in {".fasta", ".fa"}:
            unsupported_files += 1
            continue

        if stem in skipped_labels:
            dropped_files += 1
            dropped_labels.append(stem)
            continue

        dst_path = os.path.join(args.output_fasta_dir, fname)
        shutil.copy2(src_path, dst_path)
        kept_files += 1

    report_path = os.path.join(args.output_fasta_dir, "filter_report.txt")
    with open(report_path, "w") as outfile:
        outfile.write(f"input_fasta_dir: {args.input_fasta_dir}\n")
        outfile.write(f"skipped_file: {args.skipped_file}\n")
        outfile.write(f"total_skipped_labels: {len(skipped_labels)}\n")
        outfile.write(f"kept_files: {kept_files}\n")
        outfile.write(f"dropped_files: {dropped_files}\n")
        outfile.write(f"unsupported_files: {unsupported_files}\n")
        outfile.write("\n[dropped_labels]\n")
        for label in sorted(dropped_labels):
            outfile.write(f"{label}\n")

    print(f"Done. Kept {kept_files}, dropped {dropped_files}.")
    print(f"Filtered FASTA dir: {args.output_fasta_dir}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter FASTA files by labels listed in skipped_sequences.txt"
    )
    parser.add_argument(
        "input_fasta_dir",
        type=str,
        help="Directory containing FASTA files (.fasta/.fa).",
    )
    parser.add_argument(
        "skipped_file",
        type=str,
        help="Path to skipped_sequences.txt (one label per line).",
    )
    parser.add_argument(
        "output_fasta_dir",
        type=str,
        help="Directory to write filtered FASTA files.",
    )

    args = parser.parse_args()
    main(args)
