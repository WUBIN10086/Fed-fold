"""Verify FP32 LoRA adapter merging and emit a machine-readable JSON report."""

import argparse
import json
from pathlib import Path

from scripts.export_lora_checkpoint import (
    compare_export_to_base,
    extract_adapter_weights,
    load_canonical_fp32_base,
    load_checkpoint,
    merge_adapters_into_base,
    resolve_export_lora_config,
    validate_merged_state_dict,
)


def verify_roundtrip(
    adapter_checkpoint: Path,
    base_checkpoint: Path,
    base_weights_source: str,
    adapter_weights_source: str,
    config_preset: str,
    experiment_config_json: str = "",
    lora_rank: int = None,
    lora_alpha: float = None,
    lora_scale: float = 1.0,
):
    checkpoint = load_checkpoint(adapter_checkpoint)
    config = resolve_export_lora_config(
        checkpoint,
        rank=lora_rank,
        alpha=lora_alpha,
    )
    adapters = extract_adapter_weights(checkpoint, adapter_weights_source)
    base, reference = load_canonical_fp32_base(
        base_checkpoint,
        base_weights_source,
        config_preset,
        experiment_config_json,
    )
    merged, target_keys = merge_adapters_into_base(
        base,
        adapters,
        alpha=config.alpha,
        rank=config.rank,
        lora_scale=lora_scale,
    )
    zero, zero_target_keys = merge_adapters_into_base(
        base,
        adapters,
        alpha=config.alpha,
        rank=config.rank,
        lora_scale=0.0,
    )
    validate_merged_state_dict(
        merged,
        reference,
        base=base,
        target_keys=target_keys,
    )
    validate_merged_state_dict(
        zero,
        reference,
        base=base,
        target_keys=zero_target_keys,
    )
    report = compare_export_to_base(merged, base, target_keys)
    zero_report = compare_export_to_base(zero, base, zero_target_keys)
    report.update({
        "base_checkpoint": str(base_checkpoint),
        "base_source": base_weights_source,
        "adapter_checkpoint": str(adapter_checkpoint),
        "adapter_source": adapter_weights_source,
        "adapter_tensor_count": len(adapters),
        "rank": config.rank,
        "alpha": config.alpha,
        "lora_scale": float(lora_scale),
        "scale0_bitwise_identity": zero_report["bitwise_identity"],
        "dtype_distribution": {
            str(dtype): sum(tensor.dtype == dtype for tensor in merged.values())
            for dtype in sorted(
                {tensor.dtype for tensor in merged.values()},
                key=str,
            )
        },
    })
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-checkpoint", required=True, type=Path)
    parser.add_argument("--base-checkpoint", required=True, type=Path)
    parser.add_argument(
        "--base-weights-source",
        choices=("ema", "module"),
        default="ema",
    )
    parser.add_argument(
        "--adapter-weights-source",
        choices=("ema", "model"),
        default="model",
    )
    parser.add_argument(
        "--config-preset",
        default="seq_model_esm1b_ptm",
    )
    parser.add_argument("--experiment-config-json", default="")
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--lora-alpha", type=float, default=None)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    report = verify_roundtrip(
        adapter_checkpoint=args.adapter_checkpoint,
        base_checkpoint=args.base_checkpoint,
        base_weights_source=args.base_weights_source,
        adapter_weights_source=args.adapter_weights_source,
        config_preset=args.config_preset,
        experiment_config_json=args.experiment_config_json,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_scale=args.lora_scale,
    )
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(serialized + "\n")
    print(serialized)


if __name__ == "__main__":
    main()
