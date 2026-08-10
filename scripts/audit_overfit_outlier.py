#!/usr/bin/env python3
"""Audit overfit outliers (default: 9u78_A) for mapping/cache/inference integrity."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.tmscore_from_pdb import parse_tmscore_output
from scripts.lddt_ca_from_pdb import compute_lddt_ca, read_ca_structure
from scripts.lora_target_registry import get_target

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O",
}

DEFAULT_RUN = REPO / "outputs" / "fed_lora_hardcase_fed_v1"
NAMESPACE = "target_ablation_v1"
LABEL = "9u78_A"
CLUSTER = "c935"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_path(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"Cannot fingerprint empty artifact directory: {path}")
    for item in files:
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_fasta_sequence(path: Path) -> str:
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith(">"):
            continue
        lines.append(line.strip())
    return "".join(lines).upper()


def parse_pdb_chain(path: Path) -> dict:
    residues = []
    cas = []
    chains = set()
    seen = set()
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            atom = line[12:16].strip()
            resname = line[17:20].strip()
            chain = line[21].strip() or "_"
            resseq = line[22:26].strip()
            chains.add(chain)
            key = (chain, resseq, resname)
            if key not in seen and atom in {"CA", "C"}:
                # Prefer CA for residue identity; fall back first atom later.
                pass
            if atom == "CA":
                cas.append((chain, resseq, resname, float(line[30:38]), float(line[38:46]), float(line[46:54])))
                if key not in seen:
                    seen.add(key)
                    residues.append((chain, resseq, resname))
            elif key not in seen and atom == "N":
                # defer; CA preferred
                pass
    # If some residues lacked CA, still count unique residue keys from ATOM lines.
    if not residues:
        with path.open(encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if not line.startswith(("ATOM", "HETATM")):
                    continue
                resname = line[17:20].strip()
                chain = line[21].strip() or "_"
                resseq = line[22:26].strip()
                key = (chain, resseq, resname)
                if key not in seen:
                    seen.add(key)
                    residues.append(key)
    seq = "".join(AA3_TO_1.get(r[2], "X") for r in residues)
    return {
        "path": str(path),
        "sha256": sha256_file(path) if path.exists() else "",
        "chains": sorted(chains),
        "n_residues": len(residues),
        "n_ca": len(cas),
        "sequence": seq,
        "sequence_hash": hashlib.sha256(seq.encode()).hexdigest() if seq else "",
        "residue_ids_head": [f"{c}:{s}:{n}" for c, s, n in residues[:5]],
        "ca_coords": cas,
    }


def ca_rmsd(pred: dict, native: dict) -> Optional[float]:
    """RMSD over intersecting CA residues matched by (chain,resseq) when possible,
    else by residue index order if lengths match."""
    import math

    pred_map = {(c, s): (x, y, z) for c, s, _n, x, y, z in pred.get("ca_coords", [])}
    nat_map = {(c, s): (x, y, z) for c, s, _n, x, y, z in native.get("ca_coords", [])}
    keys = sorted(set(pred_map) & set(nat_map))
    pairs = []
    if keys:
        pairs = [(pred_map[k], nat_map[k]) for k in keys]
    elif len(pred.get("ca_coords", [])) == len(native.get("ca_coords", [])) and pred.get("ca_coords"):
        pairs = [
            ((a[3], a[4], a[5]), (b[3], b[4], b[5]))
            for a, b in zip(pred["ca_coords"], native["ca_coords"])
        ]
    if not pairs:
        return None
    acc = 0.0
    for (x1, y1, z1), (x2, y2, z2) in pairs:
        acc += (x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2
    return math.sqrt(acc / len(pairs))


def run_tmscore(pred: Path, native: Path, tm_exec: Path) -> dict:
    proc = subprocess.run(
        [str(tm_exec), str(pred), str(native)],
        check=True,
        capture_output=True,
        text=True,
    )
    stdout = proc.stdout
    tm1, tm2, rmsd, aligned = parse_tmscore_output(stdout)
    # Parse lengths if present
    len1 = None
    len2 = None
    m1 = re.search(r"Length of Chain_1:\s*([0-9]+)", stdout)
    m2 = re.search(r"Length of Chain_2:\s*([0-9]+)", stdout)
    if m1:
        len1 = int(m1.group(1))
    if m2:
        len2 = int(m2.group(1))
    return {
        "tm_norm_chain1": tm1,
        "tm_norm_chain2": tm2,
        "tm_selected": max(tm1, tm2),
        "rmsd": rmsd,
        "aligned_len": aligned,
        "chain1_length": len1,
        "chain2_length": len2,
        "stdout": stdout,
    }


def file_meta(path: Path) -> dict:
    if not path.exists():
        return {"path": str(path), "exists": False}
    st = path.stat()
    return {
        "path": str(path.resolve()),
        "exists": True,
        "sha256": sha256_file(path),
        "mtime": st.st_mtime,
        "size": st.st_size,
    }


def assert_unique_prediction_dirs(paths: Sequence[Path]) -> List[str]:
    errors = []
    resolved = [p.resolve() for p in paths if p.exists()]
    if len(set(resolved)) != len(resolved):
        errors.append("duplicate_prediction_directories")
    return errors


def audit_static_and_mapping(
    *,
    run_root: Path,
    label: str = LABEL,
    cluster: str = CLUSTER,
) -> dict:
    private = run_root / "clients" / "client_0" / "private"
    overfit = private / NAMESPACE / "overfit8"
    native = private / "difficulty" / "native" / f"{label}.pdb"
    fasta = overfit / "assets" / "solo_fasta_dir" / f"{label}.fasta"
    cache = json.loads((overfit / "overfit8_chain_data_cache.json").read_text())
    parent = run_root / "server" / "rounds" / "round_000" / "global_model.pt"

    mapping_errors: List[str] = []
    if label not in cache and label.upper() not in {k.upper() for k in cache}:
        mapping_errors.append("label_missing_from_overfit_chain_cache")
    cache_entry = cache.get(label) or cache.get(label.upper()) or {}
    if cache_entry.get("cluster_id") not in (None, "", cluster) and cache_entry.get("cluster_size") is None:
        # cluster may only be encoded via regrouping; tolerate missing explicit id
        pass

    fasta_seq = read_fasta_sequence(fasta) if fasta.exists() else ""
    native_info = parse_pdb_chain(native) if native.exists() else {}
    if fasta_seq and native_info.get("sequence"):
        # Allow X mismatches from nonstandard residues; require high identity.
        if fasta_seq != native_info["sequence"]:
            # soft check: length mismatch is hard error
            if len(fasta_seq) != len(native_info["sequence"]):
                mapping_errors.append(
                    f"fasta_native_length_mismatch:{len(fasta_seq)}!={native_info['n_residues']}"
                )
            else:
                mismatches = sum(a != b and b != "X" and a != "X" for a, b in zip(fasta_seq, native_info["sequence"]))
                if mismatches > max(2, len(fasta_seq) // 20):
                    mapping_errors.append(f"fasta_native_sequence_mismatch_count={mismatches}")

    if label.split("_")[-1] not in (native_info.get("chains") or [label.split("_")[-1]]):
        # single-chain natives often omit chain or use A
        chains = native_info.get("chains") or []
        if chains and label.split("_")[-1] not in chains and "A" not in chains and "_" not in chains:
            mapping_errors.append(f"native_chain_mismatch:{chains}")

    t5a_pred = (
        private
        / NAMESPACE
        / "T5A"
        / "overfit8"
        / "seed_42"
        / "eval"
        / "step_200"
        / "scale_1p0"
        / "predictions"
        / f"{label}_seq_model_esm1b_ptm_unrelaxed.pdb"
    )
    t5b_pred = (
        private
        / NAMESPACE
        / "T5B"
        / "overfit8"
        / "seed_42"
        / "eval"
        / "step_200"
        / "scale_1p0"
        / "predictions"
        / f"{label}_seq_model_esm1b_ptm_unrelaxed.pdb"
    )
    pred_dirs = [t5a_pred.parent, t5b_pred.parent]
    mapping_errors.extend(assert_unique_prediction_dirs(pred_dirs))
    if t5a_pred.exists() and t5b_pred.exists():
        if sha256_file(t5a_pred) == sha256_file(t5b_pred):
            mapping_errors.append("t5a_t5b_prediction_sha_identical_possible_reuse")

    models = {}
    for slug in ("T5A", "T5B"):
        model = (
            private
            / NAMESPACE
            / slug
            / "overfit8"
            / "seed_42"
            / "eval"
            / "step_200"
            / "scale_1p0"
            / "model_scale_1.0.pt"
        )
        manifest = model.with_suffix(model.suffix + ".export_manifest.json")
        if not manifest.exists():
            # alternate naming used by export
            alt = model.parent / "model_scale_1.0.export_manifest.json"
            manifest = alt if alt.exists() else manifest
        manifest_payload = json.loads(manifest.read_text()) if manifest.exists() else {}
        if not model.exists():
            mapping_errors.append(f"missing_exported_model:{slug}")
        if not manifest.exists():
            mapping_errors.append(f"missing_export_manifest:{slug}")
        else:
            manifest_output = manifest_payload.get("output")
            if manifest_output and Path(manifest_output).resolve() != model.resolve():
                mapping_errors.append(f"manifest_output_mismatch:{slug}")
            report = manifest_payload.get("report") or {}
            manifest_base = report.get("base_checkpoint")
            if manifest_base and Path(manifest_base).resolve() != parent.resolve():
                mapping_errors.append(f"manifest_parent_mismatch:{slug}")
            if manifest_payload.get("lora_rank") != 4:
                mapping_errors.append(f"manifest_rank_mismatch:{slug}")
            if float(manifest_payload.get("lora_alpha", -1)) != 8.0:
                mapping_errors.append(f"manifest_alpha_mismatch:{slug}")
            reported_target = manifest_payload.get("lora_target")
            if reported_target and reported_target != get_target(slug).target:
                mapping_errors.append(f"manifest_target_mismatch:{slug}")
            reported_output_sha = manifest_payload.get("output_sha256")
            if reported_output_sha and model.exists() and reported_output_sha != sha256_file(model):
                mapping_errors.append(f"manifest_model_sha_mismatch:{slug}")
            reported_parent_sha = manifest_payload.get("base_checkpoint_sha256")
            if reported_parent_sha and parent.exists() and reported_parent_sha != sha256_file(parent):
                mapping_errors.append(f"manifest_parent_sha_mismatch:{slug}")
        models[slug] = {
            "model": file_meta(model),
            "manifest": manifest_payload,
            "manifest_path": str(manifest) if manifest.exists() else "",
        }

    if not parent.exists():
        mapping_errors.append("missing_parent_global_model")
    if not t5a_pred.exists():
        mapping_errors.append("missing_t5a_prediction")
    if not t5b_pred.exists():
        mapping_errors.append("missing_t5b_prediction")

    payload = {
        "label": label,
        "expected_cluster": cluster,
        "parent_global": file_meta(parent),
        "native": native_info if native_info else file_meta(native),
        "fasta": {
            **file_meta(fasta),
            "sequence_hash": hashlib.sha256(fasta_seq.encode()).hexdigest() if fasta_seq else "",
            "length": len(fasta_seq),
        },
        "chain_cache_entry_keys": sorted(cache_entry.keys()) if isinstance(cache_entry, dict) else [],
        "predictions": {
            "T5A": {
                **file_meta(t5a_pred),
                **(parse_pdb_chain(t5a_pred) if t5a_pred.exists() else {}),
            },
            "T5B": {
                **file_meta(t5b_pred),
                **(parse_pdb_chain(t5b_pred) if t5b_pred.exists() else {}),
            },
        },
        "models": models,
        "mapping_errors": mapping_errors,
        "audit_integrity_pass": len(mapping_errors) == 0
        and native.exists()
        and fasta.exists()
        and t5a_pred.exists()
        and t5b_pred.exists()
        and parent.exists(),
    }
    # Drop bulky coords from JSON summary copies
    for key in ("T5A", "T5B"):
        payload["predictions"][key].pop("ca_coords", None)
    if "ca_coords" in payload.get("native", {}):
        payload["native"] = {k: v for k, v in payload["native"].items() if k != "ca_coords"}
    return payload


def metric_rescore(
    *,
    pred: Path,
    native: Path,
    tm_exec: Path,
    out_dir: Path,
    repeats: int = 2,
) -> dict:
    results = []
    pred_structure = read_ca_structure(pred)
    native_structure = read_ca_structure(native)
    for i in range(repeats):
        scored = run_tmscore(pred, native, tm_exec)
        lddt_score, aligned_ca, scored_pairs = compute_lddt_ca(
            pred_structure, native_structure
        )
        scored["lddt_ca"] = lddt_score
        scored["lddt_aligned_ca"] = aligned_ca
        scored["lddt_scored_pairs"] = scored_pairs
        stdout_path = out_dir / f"tmscore_repeat_{i}.txt"
        stdout_path.write_text(scored["stdout"], encoding="utf-8")
        scored_no_stdout = {k: v for k, v in scored.items() if k != "stdout"}
        scored_no_stdout["stdout_path"] = str(stdout_path)
        results.append(scored_no_stdout)
    tms = [r["tm_selected"] for r in results]
    lddts = [r["lddt_ca"] for r in results]
    reproducible = (
        max(tms) - min(tms) <= 1e-6
        and max(lddts) - min(lddts) <= 1e-9
    )
    return {
        "repeats": results,
        "metric_reproducible": reproducible,
        "tm_selected_values": tms,
        "tm_selected_mean": sum(tms) / len(tms),
        "lddt_ca_values": lddts,
        "lddt_ca_mean": sum(lddts) / len(lddts),
    }


def independent_inference_replay(
    *,
    run_root: Path,
    source_run: Path,
    out_dir: Path,
    label: str = LABEL,
    slug: str = "T5A",
    seed: int = 42,
    gpu_id: str = "0",
    python_bin: str = sys.executable,
    n_replays: int = 2,
) -> dict:
    """Re-export and re-infer label into unique dirs; compare TM and coords."""
    private = run_root / "clients" / "client_0" / "private"
    overfit = private / NAMESPACE / "overfit8"
    ckpt = (
        private
        / NAMESPACE
        / slug
        / "overfit8"
        / f"seed_{seed}"
        / "training"
        / "checkpoints"
        / "24-200.ckpt"
    )
    parent = run_root / "server" / "rounds" / "round_000" / "global_model.pt"
    native = private / "difficulty" / "native" / f"{label}.pdb"
    tm_exec = REPO / "tmscore" / "TMscore"
    assets = overfit / "assets"

    # Minimal single-label asset dirs
    label_assets = out_dir / "single_label_assets"
    for sub, src_name in (
        ("solo_fasta_dir", "solo_fasta_dir"),
        ("solo_alignment_dir", "solo_alignment_dir"),
        ("mmcif_files", "mmcif_files"),
    ):
        dst = label_assets / sub
        dst.mkdir(parents=True, exist_ok=True)
        if sub == "solo_fasta_dir":
            src = assets / src_name / f"{label}.fasta"
            link = dst / f"{label}.fasta"
            if not link.exists():
                link.symlink_to(src.resolve())
        elif sub == "solo_alignment_dir":
            src = assets / src_name / label
            link = dst / label
            if not link.exists():
                link.symlink_to(src.resolve())
        else:
            pdb_id = label.split("_")[0].lower()
            src = assets / src_name / f"{pdb_id}.cif"
            link = dst / f"{pdb_id}.cif"
            if not link.exists() and src.exists():
                link.symlink_to(src.resolve())

    replay_rows = []
    integrity_errors = []
    for i in range(n_replays):
        cand = out_dir / f"replay_{i}"
        if cand.exists():
            shutil.rmtree(cand)
        cand.mkdir(parents=True)
        model = cand / "model_scale_1.0.pt"
        env = {**dict(**{k: v for k, v in __import__("os").environ.items()}), "CUDA_VISIBLE_DEVICES": gpu_id, "PYTHONPATH": str(REPO)}
        subprocess.run(
            [
                python_bin,
                str(REPO / "scripts" / "export_lora_checkpoint.py"),
                "--input", str(ckpt),
                "--output", str(model),
                "--base-checkpoint", str(parent),
                "--base-weights-source", "auto",
                "--adapter-weights-source", "model",
                "--lora-rank", "4",
                "--lora-alpha", "8",
                "--lora-scale", "1.0",
                "--config-preset", "seq_model_esm1b_ptm",
            ],
            check=True,
            env=env,
            cwd=str(REPO),
        )
        subprocess.run(
            [
                python_bin,
                str(REPO / "run_pretrained_openfold.py"),
                str(label_assets / "solo_fasta_dir"),
                str(label_assets / "mmcif_files"),
                "--use_precomputed_alignments", str(label_assets / "solo_alignment_dir"),
                "--use_single_seq_mode",
                "--output_dir", str(cand),
                "--model_device", "cuda:0",
                "--skip_relaxation",
                "--config_preset", "seq_model_esm1b_ptm",
                "--openfold_checkpoint_path", str(model),
                "--checkpoint_weights_source", "auto",
                "--data_random_seed", str(seed),
                "--precision", "fp32",
            ],
            check=True,
            env=env,
            cwd=str(REPO),
        )
        manifest_path = model.with_suffix(".export_manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checks = {
            "output_sha256": sha256_file(model),
            "base_checkpoint_sha256": sha256_file(parent),
            "adapter_checkpoint_sha256": sha256_path(ckpt),
            "lora_target": get_target(slug).target,
            "lora_rank": 4,
            "lora_alpha": 8.0,
            "lora_scale": 1.0,
            "scale_semantics": "unit_raw_delta",
        }
        for key, expected in checks.items():
            if manifest.get(key) != expected:
                integrity_errors.append(f"replay_{i}_{key}_mismatch")
        if not manifest.get("lora_schema_fingerprint"):
            integrity_errors.append(f"replay_{i}_schema_fingerprint_missing")
        if not manifest.get("adapter_shapes") or not manifest.get("adapter_dtypes"):
            integrity_errors.append(f"replay_{i}_adapter_schema_missing")

        pred = cand / "predictions" / f"{label}_seq_model_esm1b_ptm_unrelaxed.pdb"
        scored = run_tmscore(pred, native, tm_exec)
        pred_info = parse_pdb_chain(pred)
        native_info = parse_pdb_chain(native)
        lddt_score, aligned_ca, scored_pairs = compute_lddt_ca(
            read_ca_structure(pred), read_ca_structure(native)
        )
        replay_rows.append(
            {
                "replay_index": i,
                "prediction_dir": str(cand / "predictions"),
                "prediction_sha256": sha256_file(pred),
                "model_sha256": sha256_file(model),
                "model_manifest": str(manifest_path),
                "lora_schema_fingerprint": manifest.get("lora_schema_fingerprint"),
                "tm_selected": scored["tm_selected"],
                "lddt_ca": lddt_score,
                "lddt_aligned_ca": aligned_ca,
                "lddt_scored_pairs": scored_pairs,
                "ca_rmsd_vs_native": ca_rmsd(pred_info, native_info),
                "n_residues": pred_info["n_residues"],
                "n_ca": pred_info["n_ca"],
            }
        )

    tms = [r["tm_selected"] for r in replay_rows]
    lddts = [r["lddt_ca"] for r in replay_rows]
    shas = [r["prediction_sha256"] for r in replay_rows]
    inference_reproducible = (max(tms) - min(tms) <= 1e-4) if tms else False
    replay_pairwise_ca_rmsd = None
    if len(replay_rows) >= 2:
        first = Path(replay_rows[0]["prediction_dir"]) / f"{label}_seq_model_esm1b_ptm_unrelaxed.pdb"
        second = Path(replay_rows[1]["prediction_dir"]) / f"{label}_seq_model_esm1b_ptm_unrelaxed.pdb"
        replay_pairwise_ca_rmsd = ca_rmsd(parse_pdb_chain(first), parse_pdb_chain(second))
        inference_reproducible = bool(
            inference_reproducible
            and max(lddts) - min(lddts) <= 1e-6
            and replay_pairwise_ca_rmsd is not None
            and replay_pairwise_ca_rmsd <= 1e-4
        )
    return {
        "slug": slug,
        "seed": seed,
        "checkpoint": str(ckpt),
        "replays": replay_rows,
        "tm_values": tms,
        "lddt_values": lddts,
        "prediction_shas": shas,
        "replay_pairwise_ca_rmsd": replay_pairwise_ca_rmsd,
        "inference_reproducible": inference_reproducible,
        "cache_reuse_detected": len(set(r["prediction_dir"] for r in replay_rows)) != len(replay_rows),
        "integrity_errors": integrity_errors,
    }


def build_audit_summary(
    static: dict,
    rescore: dict,
    replay: Optional[dict] = None,
) -> dict:
    mapping_errors = list(static.get("mapping_errors") or [])
    integrity = bool(static.get("audit_integrity_pass"))
    if replay and replay.get("cache_reuse_detected"):
        mapping_errors.append("cache_reuse_detected")
        integrity = False
    if replay and replay.get("integrity_errors"):
        mapping_errors.extend(replay["integrity_errors"])
        integrity = False
    metric_ok = bool(rescore.get("metric_reproducible"))
    replay_complete = replay is not None
    inference_ok = bool(replay.get("inference_reproducible")) if replay else False
    audit_complete = bool(integrity and metric_ok and replay_complete)
    audit_pass = bool(audit_complete and inference_ok)
    return {
        "label": static.get("label", LABEL),
        "audit_integrity_pass": integrity,
        "metric_reproducible": bool(rescore.get("metric_reproducible")),
        "inference_reproducible": (
            None if replay is None else bool(replay.get("inference_reproducible"))
        ),
        "cache_reuse_detected": bool(replay.get("cache_reuse_detected")) if replay else False,
        "mapping_errors": mapping_errors,
        "tm_rescore_mean": rescore.get("tm_selected_mean"),
        "replay_tm_values": None if replay is None else replay.get("tm_values"),
        "replay_lddt_values": None if replay is None else replay.get("lddt_values"),
        "replay_pairwise_ca_rmsd": (
            None if replay is None else replay.get("replay_pairwise_ca_rmsd")
        ),
        "audit_complete": audit_complete,
        "audit_pass": audit_pass,
        "note": (
            "metric_reproducible only proves scoring determinism; "
            "inference_reproducible requires independent export/infer replays"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--label", default=LABEL)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument(
        "--python-bin",
        default=str(Path.home() / "miniconda3" / "envs" / "fedfold" / "bin" / "python"),
    )
    args = parser.parse_args()
    out_dir = args.out_dir or (
        args.run_root
        / "clients"
        / "client_0"
        / "private"
        / NAMESPACE
        / "diagnostics"
        / args.label
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    static = audit_static_and_mapping(run_root=args.run_root, label=args.label)
    write_json(out_dir / "static_integrity.json", static)

    private = args.run_root / "clients" / "client_0" / "private"
    pred = (
        private
        / NAMESPACE
        / "T5A"
        / "overfit8"
        / "seed_42"
        / "eval"
        / "step_200"
        / "scale_1p0"
        / "predictions"
        / f"{args.label}_seq_model_esm1b_ptm_unrelaxed.pdb"
    )
    native = private / "difficulty" / "native" / f"{args.label}.pdb"
    rescore = metric_rescore(
        pred=pred,
        native=native,
        tm_exec=REPO / "tmscore" / "TMscore",
        out_dir=out_dir,
        repeats=2,
    )
    write_json(out_dir / "tm_rescore.json", rescore)

    replay = None
    if not args.skip_inference:
        replay = independent_inference_replay(
            run_root=args.run_root,
            source_run=REPO / "outputs" / "fed_lora_fp32",
            out_dir=out_dir / "independent_inference",
            label=args.label,
            gpu_id=args.gpu_id,
            python_bin=args.python_bin if Path(args.python_bin).exists() else sys.executable,
        )
        write_json(out_dir / "independent_inference.json", replay)

    summary = build_audit_summary(static, rescore, replay)
    write_json(out_dir / "audit_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["audit_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
