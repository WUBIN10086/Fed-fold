import argparse
import logging
import os
import string
from collections import defaultdict
from openfold.data import mmcif_parsing
from openfold.np import protein, residue_constants


def main(args):
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Reading CIFs from: {args.data_dir}")
    print(f"Writing FASTAs to: {args.output_dir}")

    for fname in os.listdir(args.data_dir):
        basename, ext = os.path.splitext(fname)
        #basename = basename.upper()  # e.g. 2GAI
        fpath = os.path.join(args.data_dir, fname)

        if (ext == ".cif"):
            with open(fpath, 'r') as fp:
                mmcif_str = fp.read()

            # 解析 CIF
            mmcif = mmcif_parsing.parse(
                file_id=basename, mmcif_string=mmcif_str
            )
            if (mmcif.mmcif_object is None):
                logging.warning(f'Failed to parse {fname}...')
                continue

            mmcif = mmcif.mmcif_object
            # 遍历每一个链，单独保存文件
            for chain, seq in mmcif.chain_to_seqres.items():
                # 构造正确的 ID：PDBID_CHAIN (例如 2GAI_A)
                chain_id = '_'.join([basename, chain])

                # 输出文件名：2GAI_A.fasta
                output_filename = os.path.join(args.output_dir, f"{chain_id}.fasta")

                with open(output_filename, "w") as fp:
                    fp.write(f">{chain_id}\n")
                    fp.write(seq + "\n")

                print(f"Generated: {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir", type=str,
        help="Path to a directory containing mmCIF files (e.g. train_subset)"
    )
    parser.add_argument(
        "output_dir", type=str,
        help="Path to output FASTA directory (e.g. solo_fasta_dir)"
    )

    args = parser.parse_args()
    main(args)