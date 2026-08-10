#!/usr/bin/env python3
"""Aggregate client raw LoRA updates into a global pure-weight checkpoint.

MVP aggregation:
  delta_W_k = (alpha / rank) * B_k @ A_k
  delta_W   = sum_k w_k * delta_W_k
  G_{t+1}   = G_t + delta_W

Rejects EMA adapters, mismatched parents/configs, and scaled adapters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.export_lora_checkpoint import (  # noqa: E402
    extract_adapter_weights,
    load_checkpoint,
    resolve_export_lora_config,
)
from scripts.lora_schema import (  # noqa: E402
    assert_same_schema,
    fingerprint_from_adapters,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pure_weights(path: Path) -> Dict[str, torch.Tensor]:
    obj = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(obj, Mapping):
        raise ValueError(f"Expected a state dict at {path}")
    tensors = {
        key: value
        for key, value in obj.items()
        if isinstance(key, str) and torch.is_tensor(value)
    }
    if not tensors:
        raise ValueError(f"No tensors found in {path}")
    # Reject accidental LoRA-bearing pure dicts for parent globals.
    if any(".lora_" in key for key in tensors):
        raise ValueError(
            f"Parent/global weights must be merged full-model tensors without "
            f"lora_A/B keys: {path}"
        )
    return {
        key: (
            value.detach().cpu().float().clone()
            if torch.is_floating_point(value)
            else value.detach().cpu().clone()
        )
        for key, value in tensors.items()
    }


def compute_effective_deltas(
    adapters: Mapping[str, torch.Tensor],
    alpha: float,
    rank: int,
    scale: float = 1.0,
) -> Dict[str, torch.Tensor]:
    if abs(float(scale) - 1.0) > 1e-12:
        raise ValueError(
            "FedLoRA MVP rejects non-unit aggregation scale; got "
            f"scale={scale}"
        )
    prefixes = sorted(
        key[: -len(".lora_A")]
        for key in adapters
        if key.endswith(".lora_A")
    )
    deltas = {}
    for prefix in prefixes:
        a = adapters[prefix + ".lora_A"].detach().cpu().float()
        b = adapters[prefix + ".lora_B"].detach().cpu().float()
        weight_key = prefix + ".weight"
        expected = (rank, a.shape[1] if a.ndim == 2 else None)
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != rank or b.shape[1] != rank:
            raise ValueError(
                f"Unexpected LoRA shapes for {prefix}: A={tuple(a.shape)} "
                f"B={tuple(b.shape)} rank={rank}"
            )
        deltas[weight_key] = (float(alpha) / float(rank)) * torch.matmul(b, a)
    if not deltas:
        raise ValueError("No LoRA adapters found")
    return deltas


def frobenius_norm(deltas: Mapping[str, torch.Tensor]) -> float:
    total = 0.0
    for tensor in deltas.values():
        total += float(torch.sum(tensor.float() * tensor.float()).item())
    return total ** 0.5


def clip_deltas(
    deltas: Mapping[str, torch.Tensor],
    max_norm: Optional[float],
) -> Tuple[Dict[str, torch.Tensor], float, bool]:
    norm = frobenius_norm(deltas)
    if max_norm is None or max_norm <= 0 or norm <= max_norm:
        return {key: value.clone() for key, value in deltas.items()}, norm, False
    scale = max_norm / (norm + 1e-12)
    clipped = {key: value * scale for key, value in deltas.items()}
    return clipped, frobenius_norm(clipped), True


def resolve_parent_weight_key(
    adapter_weight_key: str,
    parent: Mapping[str, torch.Tensor],
) -> str:
    """Resolve LoRA module aliases to canonical OpenFold state-dict keys."""
    candidates = [adapter_weight_key]

    block_match = re.match(
        r"^(evoformer\.blocks\.\d+)\.(.+)$", adapter_weight_key
    )
    if block_match:
        block, remainder = block_match.groups()
        candidates.append(f"{block}.core.{remainder}")
        if remainder.startswith("pair_stack."):
            candidates.append(
                f"{block}.core.{remainder[len('pair_stack.'):]}"
            )

    point_projection_match = re.match(
        r"^(structure_module\.ipa\.linear_(?:q|kv)_points)\.linear\.weight$",
        adapter_weight_key,
    )
    if point_projection_match:
        candidates.append(f"{point_projection_match.group(1)}.weight")

    matches = [key for key in candidates if key in parent]
    if not matches:
        raise ValueError(
            f"Parent is missing LoRA target weight {adapter_weight_key}"
        )
    if len(set(matches)) != 1:
        raise ValueError(
            "LoRA target alias is ambiguous for "
            f"{adapter_weight_key}: {sorted(set(matches))}"
        )
    return matches[0]


def aggregate_effective_deltas(
    parent: Mapping[str, torch.Tensor],
    client_deltas: Sequence[Mapping[str, torch.Tensor]],
    weights: Sequence[float],
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    if len(client_deltas) != len(weights):
        raise ValueError("weights length must match client updates")
    if not client_deltas:
        raise ValueError("No client updates provided")
    wsum = float(sum(weights))
    if wsum <= 0:
        raise ValueError("Aggregation weights must sum to a positive value")

    adapter_target_keys = sorted(client_deltas[0].keys())
    for idx, deltas in enumerate(client_deltas[1:], start=1):
        if sorted(deltas.keys()) != adapter_target_keys:
            raise ValueError(
                f"Client #{idx} LoRA target keys differ from client #0"
            )

    child = {
        key: (
            value.detach().cpu().float().clone()
            if torch.is_floating_point(value)
            else value.detach().cpu().clone()
        )
        for key, value in parent.items()
    }
    resolved_targets = {
        key: resolve_parent_weight_key(key, child)
        for key in adapter_target_keys
    }
    if len(set(resolved_targets.values())) != len(resolved_targets):
        raise ValueError("Multiple LoRA adapter targets resolve to one parent weight")

    for adapter_key in adapter_target_keys:
        parent_key = resolved_targets[adapter_key]
        acc = torch.zeros_like(child[parent_key], dtype=torch.float64)
        for weight, deltas in zip(weights, client_deltas):
            delta = deltas[adapter_key]
            if tuple(delta.shape) != tuple(child[parent_key].shape):
                raise ValueError(
                    f"LoRA update shape mismatch for {adapter_key} -> {parent_key}: "
                    f"delta={tuple(delta.shape)} parent={tuple(child[parent_key].shape)}"
                )
            acc += float(weight) * delta.to(torch.float64)
        child[parent_key] = child[parent_key].float() + (
            acc / wsum
        ).to(child[parent_key].dtype)
    target_keys = sorted(resolved_targets.values())
    return child, target_keys


def non_target_max_abs_diff(
    parent: Mapping[str, torch.Tensor],
    child: Mapping[str, torch.Tensor],
    target_keys: Sequence[str],
) -> float:
    target = set(target_keys)
    max_diff = 0.0
    for key, value in parent.items():
        if key in target:
            continue
        if key not in child:
            raise ValueError(f"Child missing non-target key {key}")
        if torch.is_floating_point(value):
            diff = float((child[key].float() - value.float()).abs().max().item())
            max_diff = max(max_diff, diff)
        elif not torch.equal(child[key], value):
            raise ValueError(f"Non-float non-target key changed: {key}")
    return max_diff


def secure_aggregate_placeholder(
    client_deltas: Sequence[Mapping[str, torch.Tensor]],
    weights: Sequence[float],
) -> Mapping[str, torch.Tensor]:
    """Interface hook for future secure aggregation.

    Current trusted-server MVP simply returns weighted plaintext average.
    """
    wsum = float(sum(weights))
    keys = sorted(client_deltas[0].keys())
    out = {}
    for key in keys:
        acc = torch.zeros_like(client_deltas[0][key], dtype=torch.float64)
        for weight, deltas in zip(weights, client_deltas):
            acc += float(weight) * deltas[key].to(torch.float64)
        out[key] = (acc / wsum).to(client_deltas[0][key].dtype)
    return out


def aggregate_round(
    parent_path: Path,
    client_checkpoints: Sequence[Path],
    weights: Sequence[float],
    adapter_source: str = "model",
    max_update_norm: Optional[float] = None,
    rank: Optional[int] = None,
    alpha: Optional[float] = None,
    expected_parent_sha: Optional[str] = None,
    round_id: Optional[int] = None,
    client_ids: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, torch.Tensor], dict]:
    if adapter_source != "model":
        raise ValueError(
            "FedLoRA MVP requires adapter_source='model' (raw), got "
            f"{adapter_source!r}"
        )
    parent_sha = sha256_file(parent_path)
    if expected_parent_sha and expected_parent_sha != parent_sha:
        raise ValueError(
            f"Parent SHA mismatch: expected {expected_parent_sha}, got {parent_sha}"
        )
    parent = load_pure_weights(parent_path)

    client_ids = list(client_ids or [f"client_{i}" for i in range(len(client_checkpoints))])
    if len(client_ids) != len(client_checkpoints):
        raise ValueError("client_ids length must match checkpoints")

    deltas_list = []
    update_records = []
    shared_rank = rank
    shared_alpha = alpha
    shared_target = None
    schema_fps = []

    for client_id, ckpt in zip(client_ids, client_checkpoints):
        checkpoint = load_checkpoint(Path(ckpt))
        lora_cfg = resolve_export_lora_config(checkpoint, rank=rank, alpha=alpha)
        if shared_rank is None:
            shared_rank = lora_cfg.rank
            shared_alpha = lora_cfg.alpha
            shared_target = lora_cfg.target
        else:
            if lora_cfg.rank != shared_rank or float(lora_cfg.alpha) != float(shared_alpha):
                raise ValueError(
                    f"{client_id} LoRA rank/alpha mismatch with peer clients"
                )
            if shared_target is not None and lora_cfg.target != shared_target:
                raise ValueError(
                    f"{client_id} LoRA target mismatch with peer clients"
                )
            shared_target = lora_cfg.target

        # Reject EMA by only extracting from raw model source.
        adapters = extract_adapter_weights(checkpoint, source="model")
        fp = fingerprint_from_adapters(
            adapters,
            target=str(lora_cfg.target),
            rank=int(lora_cfg.rank),
            alpha=float(lora_cfg.alpha),
            parent_sha256=parent_sha,
        )
        schema_fps.append((client_id, fp))
        deltas = compute_effective_deltas(
            adapters,
            alpha=float(lora_cfg.alpha),
            rank=int(lora_cfg.rank),
            scale=1.0,
        )
        before_norm = frobenius_norm(deltas)
        clipped, after_norm, was_clipped = clip_deltas(deltas, max_update_norm)
        deltas_list.append(clipped)
        update_records.append(
            {
                "client_id": client_id,
                "checkpoint": str(ckpt),
                "checkpoint_sha256": sha256_file(Path(ckpt)) if Path(ckpt).is_file() else None,
                "n_weight": None,
                "update_norm_before_clip": before_norm,
                "update_norm_after_clip": after_norm,
                "clipped": was_clipped,
                "rank": int(lora_cfg.rank),
                "alpha": float(lora_cfg.alpha),
                "target": lora_cfg.target,
                "adapter_source": "model",
                "lora_schema_fingerprint": fp,
            }
        )

    assert_same_schema(schema_fps)

    # Optional secure aggregation hook (trusted average today).
    _ = secure_aggregate_placeholder(deltas_list, weights)
    child, target_keys = aggregate_effective_deltas(parent, deltas_list, weights)
    max_diff = non_target_max_abs_diff(parent, child, target_keys)

    for record, weight in zip(update_records, weights):
        record["n_weight"] = float(weight)
        record["normalized_weight"] = float(weight) / float(sum(weights))

    manifest = {
        "round_id": round_id,
        "parent_global_sha": parent_sha,
        "parent_path": str(parent_path),
        "participating_clients": client_ids,
        "client_updates": update_records,
        "normalized_weights": [
            float(w) / float(sum(weights)) for w in weights
        ],
        "target_keys": target_keys,
        "rank": shared_rank,
        "alpha": shared_alpha,
        "target": shared_target,
        "lora_schema_fingerprint": schema_fps[0][1] if schema_fps else None,
        "raw_adapter_source": "model",
        "aggregation_scale": 1.0,
        "aggregation_method": "sample_weighted_effective_delta",
        "max_update_norm": max_update_norm,
        "non_target_max_abs_diff": max_diff,
        "status": "ok" if max_diff == 0.0 else "warning_non_target_drift",
    }
    return child, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", nargs="+", type=float, default=None)
    parser.add_argument("--client-ids", nargs="+", default=None)
    parser.add_argument("--adapter-source", default="model", choices=["model"])
    parser.add_argument("--max-update-norm", type=float, default=None)
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--lora-alpha", type=float, default=None)
    parser.add_argument("--expected-parent-sha", default=None)
    parser.add_argument("--round-id", type=int, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    n = len(args.checkpoints)
    weights = args.weights if args.weights is not None else [1.0] * n
    if len(weights) != n:
        raise SystemExit("--weights length must match --checkpoints")

    child, manifest = aggregate_round(
        parent_path=args.parent,
        client_checkpoints=args.checkpoints,
        weights=weights,
        adapter_source=args.adapter_source,
        max_update_norm=args.max_update_norm,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        expected_parent_sha=args.expected_parent_sha,
        round_id=args.round_id,
        client_ids=args.client_ids,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(child, str(args.output))
    manifest["child_global_sha"] = sha256_file(args.output)
    manifest["child_path"] = str(args.output)
    manifest_path = args.manifest or args.output.with_suffix(".aggregation.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.output}")
    print(f"Wrote {manifest_path}")
    print(
        f"non_target_max_abs_diff={manifest['non_target_max_abs_diff']} "
        f"status={manifest['status']}"
    )


if __name__ == "__main__":
    main()
