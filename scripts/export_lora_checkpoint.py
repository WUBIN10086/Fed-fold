"""Merge an OpenFold LoRA checkpoint into a plain AlphaFold state dict."""

import argparse
import json
import tempfile
import warnings
from pathlib import Path
from typing import Dict, Mapping, Set, Tuple

import torch
from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)

from openfold.config import model_config
from openfold.model.model import AlphaFold
from openfold.utils.import_weights import import_openfold_weights_
from openfold.utils.lora import LoRAConfig
from openfold.utils.training_utils import extract_alphafold_weights


def load_checkpoint(path: Path) -> Mapping[str, object]:
    if path.is_dir():
        with tempfile.TemporaryDirectory() as temp_dir:
            converted = Path(temp_dir) / "converted.pt"
            convert_zero_checkpoint_to_fp32_state_dict(str(path), str(converted))
            return torch.load(
                str(converted),
                map_location="cpu",
                weights_only=False,
            )
    return torch.load(str(path), map_location="cpu", weights_only=False)


def _strip_model_prefix(
    state_dict: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    normalized = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("model."):
            key = key[len("model."):]
        elif key.startswith("loss."):
            continue
        normalized[key] = value
    return normalized


def extract_checkpoint_weights(
    checkpoint: Mapping[str, object],
    source: str,
) -> Dict[str, torch.Tensor]:
    if source == "ema":
        ema = checkpoint.get("ema")
        if not isinstance(ema, Mapping) or not isinstance(
            ema.get("params"), Mapping
        ):
            raise ValueError("Checkpoint does not contain ema.params")
        return _strip_model_prefix(ema["params"])

    if source != "model":
        raise ValueError(f"Unknown weights source: {source}")
    if isinstance(checkpoint.get("state_dict"), Mapping):
        return _strip_model_prefix(checkpoint["state_dict"])
    if isinstance(checkpoint.get("module"), Mapping):
        return _strip_model_prefix(checkpoint["module"])

    tensors = {
        key: value
        for key, value in checkpoint.items()
        if isinstance(key, str) and isinstance(value, torch.Tensor)
    }
    if not tensors:
        raise ValueError("Checkpoint does not contain raw model weights")
    return _strip_model_prefix(tensors)


def extract_adapter_weights(
    checkpoint: Mapping[str, object],
    source: str,
) -> Dict[str, torch.Tensor]:
    weights = extract_checkpoint_weights(checkpoint, source)
    adapters = {
        key: value
        for key, value in weights.items()
        if key.endswith(".lora_A") or key.endswith(".lora_B")
    }
    a_prefixes = {
        key[:-len(".lora_A")]
        for key in adapters
        if key.endswith(".lora_A")
    }
    b_prefixes = {
        key[:-len(".lora_B")]
        for key in adapters
        if key.endswith(".lora_B")
    }
    if not a_prefixes and not b_prefixes:
        raise ValueError(
            f"Checkpoint source {source!r} contains no LoRA adapters"
        )
    if a_prefixes != b_prefixes:
        raise ValueError(
            "Unpaired LoRA adapters; "
            f"missing_A={sorted(b_prefixes - a_prefixes)[:5]}, "
            f"missing_B={sorted(a_prefixes - b_prefixes)[:5]}"
        )
    return adapters


def resolve_export_lora_config(
    checkpoint: Mapping[str, object],
    rank: int = None,
    alpha: float = None,
) -> LoRAConfig:
    stored = checkpoint.get("lora_config")
    if isinstance(stored, Mapping):
        config = LoRAConfig.from_dict(stored)
        if rank is not None and rank != config.rank:
            raise ValueError(
                f"--lora-rank {rank} does not match checkpoint rank {config.rank}"
            )
        if alpha is not None and alpha != config.alpha:
            raise ValueError(
                f"--lora-alpha {alpha} does not match checkpoint alpha "
                f"{config.alpha}"
            )
        return config
    if rank is None or alpha is None:
        raise ValueError(
            "Checkpoint has no lora_config; provide --lora-rank and --lora-alpha"
        )
    return LoRAConfig(rank=rank, alpha=alpha)


def load_canonical_fp32_base(
    checkpoint_path: Path,
    weights_source: str,
    config_preset: str,
    experiment_config_json: str = "",
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    checkpoint = load_checkpoint(checkpoint_path)
    source_weights = extract_alphafold_weights(
        checkpoint,
        source=weights_source,
    )
    config = model_config(config_preset, train=False, low_prec=False)
    if experiment_config_json:
        with open(experiment_config_json, "r") as config_file:
            config.update_from_flattened_dict(json.load(config_file))
    model = AlphaFold(config).float()
    import_openfold_weights_(model=model, state_dict=source_weights)
    reference = model.state_dict()
    base = {
        key: (
            value.detach().cpu().float().clone()
            if torch.is_floating_point(value)
            else value.detach().cpu().clone()
        )
        for key, value in reference.items()
    }
    return base, reference


def merge_adapters_into_base(
    base: Mapping[str, torch.Tensor],
    adapters: Mapping[str, torch.Tensor],
    alpha: float,
    rank: int,
    lora_scale: float = 1.0,
) -> Tuple[Dict[str, torch.Tensor], Set[str]]:
    if rank <= 0:
        raise ValueError("rank must be positive")
    merged = {key: value.clone() for key, value in base.items()}
    prefixes = sorted(
        key[:-len(".lora_A")]
        for key in adapters
        if key.endswith(".lora_A")
    )
    target_keys = set()
    for prefix in prefixes:
        a_key = prefix + ".lora_A"
        b_key = prefix + ".lora_B"
        if b_key not in adapters:
            raise ValueError(f"Missing paired adapter {b_key}")
        weight_key = prefix + ".weight"
        if weight_key not in merged:
            raise ValueError(f"Adapter target has no base weight: {weight_key}")
        a = adapters[a_key].detach().cpu().float()
        b = adapters[b_key].detach().cpu().float()
        weight = merged[weight_key]
        expected = (rank, weight.shape[1]), (weight.shape[0], rank)
        if tuple(a.shape) != expected[0] or tuple(b.shape) != expected[1]:
            raise ValueError(
                f"Invalid LoRA shapes for {prefix}: A={tuple(a.shape)}, "
                f"B={tuple(b.shape)}, expected={expected}"
            )
        if lora_scale != 0:
            delta = torch.matmul(b, a)
            merged[weight_key] = (
                weight.float()
                + float(lora_scale) * (float(alpha) / rank) * delta
            )
        target_keys.add(weight_key)
    if not target_keys:
        raise ValueError("No paired LoRA adapters were found")
    return merged, target_keys


def compare_export_to_base(
    exported: Mapping[str, torch.Tensor],
    base: Mapping[str, torch.Tensor],
    target_keys: Set[str],
) -> Dict[str, object]:
    changed_keys = []
    non_target_max_abs_diff = 0.0
    for key in base:
        if not torch.equal(exported[key], base[key]):
            changed_keys.append(key)
            if key not in target_keys:
                if torch.is_floating_point(exported[key]):
                    diff = (
                        exported[key].float() - base[key].float()
                    ).abs().max().item()
                    non_target_max_abs_diff = max(
                        non_target_max_abs_diff,
                        diff,
                    )
                else:
                    non_target_max_abs_diff = float("inf")
    unexpected = sorted(set(changed_keys) - target_keys)
    return {
        "tensor_count": len(exported),
        "target_keys": sorted(target_keys),
        "changed_keys": sorted(changed_keys),
        "unexpected_changed_keys": unexpected,
        "non_target_max_abs_diff": non_target_max_abs_diff,
        "bitwise_identity": not changed_keys,
    }


def validate_merged_state_dict(
    merged: Mapping[str, torch.Tensor],
    reference: Mapping[str, torch.Tensor],
    base: Mapping[str, torch.Tensor] = None,
    target_keys: Set[str] = None,
) -> None:
    merged_keys = set(merged)
    reference_keys = set(reference)
    missing = sorted(reference_keys - merged_keys)
    extra = sorted(merged_keys - reference_keys)
    if missing or extra:
        raise ValueError(
            f"Merged keys do not match AlphaFold; missing={missing[:10]}, "
            f"extra={extra[:10]}"
        )
    for key, tensor in merged.items():
        if tensor.shape != reference[key].shape:
            raise ValueError(
                f"Shape mismatch for {key}: {tuple(tensor.shape)} vs "
                f"{tuple(reference[key].shape)}"
            )
        if torch.is_floating_point(tensor) and not torch.isfinite(tensor).all():
            raise ValueError(f"NaN or Inf detected in {key}")
        if torch.is_floating_point(tensor) and tensor.dtype != torch.float32:
            raise ValueError(f"Floating tensor is not FP32: {key}={tensor.dtype}")
    lora_keys = [key for key in merged if ".lora_" in key]
    if lora_keys:
        raise ValueError(f"LoRA keys remain after merge: {lora_keys[:10]}")
    if base is not None and target_keys is not None:
        report = compare_export_to_base(merged, base, target_keys)
        if report["unexpected_changed_keys"]:
            raise ValueError(
                "Non-target tensors changed: "
                f"{report['unexpected_changed_keys'][:10]}"
            )


def export_lora_checkpoint(
    input_path: Path,
    output_path: Path,
    base_checkpoint_path: Path,
    base_weights_source: str,
    adapter_weights_source: str,
    config_preset: str,
    experiment_config_json: str = "",
    lora_rank: int = None,
    lora_alpha: float = None,
    lora_scale: float = 1.0,
    overwrite: bool = False,
) -> Tuple[int, LoRAConfig, Dict[str, object]]:
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if not base_checkpoint_path.exists():
        raise FileNotFoundError(base_checkpoint_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Output path must differ from input checkpoint")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}; pass --overwrite to replace"
        )

    checkpoint = load_checkpoint(input_path)
    lora_config = resolve_export_lora_config(
        checkpoint,
        rank=lora_rank,
        alpha=lora_alpha,
    )
    if not lora_config.enabled:
        raise ValueError("Checkpoint LoRA rank is 0; there is nothing to merge")
    adapters = extract_adapter_weights(
        checkpoint,
        adapter_weights_source,
    )
    base, reference = load_canonical_fp32_base(
        checkpoint_path=base_checkpoint_path,
        weights_source=base_weights_source,
        config_preset=config_preset,
        experiment_config_json=experiment_config_json,
    )
    merged, target_keys = merge_adapters_into_base(
        base,
        adapters,
        alpha=lora_config.alpha,
        rank=lora_config.rank,
        lora_scale=lora_scale,
    )
    validate_merged_state_dict(
        merged,
        reference,
        base=base,
        target_keys=target_keys,
    )
    report = compare_export_to_base(merged, base, target_keys)
    report.update({
        "base_checkpoint": str(base_checkpoint_path),
        "base_weights_source": base_weights_source,
        "adapter_checkpoint": str(input_path),
        "adapter_weights_source": adapter_weights_source,
        "lora_scale": float(lora_scale),
        "dtype_distribution": {
            str(dtype): sum(
                tensor.dtype == dtype for tensor in merged.values()
            )
            for dtype in sorted(
                {tensor.dtype for tensor in merged.values()},
                key=str,
            )
        },
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, str(output_path))
    return len(merged), lora_config, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--base-checkpoint",
        required=True,
        type=Path,
        help="Explicit FP32 base checkpoint. Frozen training weights are ignored.",
    )
    parser.add_argument(
        "--base-weights-source",
        choices=("ema", "module"),
        default="ema",
    )
    parser.add_argument(
        "--adapter-weights-source",
        choices=("ema", "model"),
        default=None,
    )
    parser.add_argument(
        "--weights-source",
        choices=("ema", "model"),
        default=None,
        help="Deprecated alias for --adapter-weights-source.",
    )
    parser.add_argument(
        "--config-preset",
        default="seq_model_esm1b_ptm",
    )
    parser.add_argument("--experiment-config-json", default="")
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--lora-alpha", type=float, default=None)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args()

    if args.adapter_weights_source and args.weights_source:
        parser.error(
            "Use only one of --adapter-weights-source and --weights-source"
        )
    adapter_source = (
        args.adapter_weights_source or args.weights_source or "ema"
    )
    if args.weights_source is not None:
        warnings.warn(
            "--weights-source is deprecated; use --adapter-weights-source",
            DeprecationWarning,
        )

    count, config, report = export_lora_checkpoint(
        input_path=args.input,
        output_path=args.output,
        base_checkpoint_path=args.base_checkpoint,
        base_weights_source=args.base_weights_source,
        adapter_weights_source=adapter_source,
        config_preset=args.config_preset,
        experiment_config_json=args.experiment_config_json,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_scale=args.lora_scale,
        overwrite=args.overwrite,
    )
    print(
        f"Exported {count} FP32-base AlphaFold tensors using "
        f"{adapter_source} adapters with LoRA rank={config.rank}, "
        f"alpha={config.alpha}, scale={args.lora_scale} to {args.output}"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
