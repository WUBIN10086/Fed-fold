#!/usr/bin/env python3
"""Stable LoRA schema fingerprint for FedLoRA compatibility checks."""

from __future__ import annotations

import hashlib
import json
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch


def adapter_key_shapes(adapters: Mapping[str, torch.Tensor]) -> Dict[str, list]:
    return {
        key: list(value.shape)
        for key, value in sorted(adapters.items())
        if key.endswith(".lora_A") or key.endswith(".lora_B")
    }


def adapter_key_dtypes(adapters: Mapping[str, torch.Tensor]) -> Dict[str, str]:
    return {
        key: str(value.dtype).removeprefix("torch.")
        for key, value in sorted(adapters.items())
        if key.endswith(".lora_A") or key.endswith(".lora_B")
    }


def lora_schema_fingerprint(
    *,
    target: str,
    rank: int,
    alpha: float,
    parent_sha256: str,
    adapter_shapes: Mapping[str, Sequence[int]],
    dtype: str = "float32",
    adapter_dtypes: Optional[Mapping[str, str]] = None,
    scale_semantics: str = "unit_raw_delta",
) -> str:
    payload = {
        "target": target,
        "rank": int(rank),
        "alpha": float(alpha),
        "parent_sha256": parent_sha256,
        "adapter_shapes": {k: list(v) for k, v in sorted(adapter_shapes.items())},
        "dtype": dtype,
        "adapter_dtypes": (
            {k: str(v) for k, v in sorted(adapter_dtypes.items())}
            if adapter_dtypes is not None
            else None
        ),
        "scale_semantics": scale_semantics,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def fingerprint_from_adapters(
    adapters: Mapping[str, torch.Tensor],
    *,
    target: str,
    rank: int,
    alpha: float,
    parent_sha256: str,
) -> str:
    return lora_schema_fingerprint(
        target=target,
        rank=rank,
        alpha=alpha,
        parent_sha256=parent_sha256,
        adapter_shapes=adapter_key_shapes(adapters),
        dtype="per_tensor",
        adapter_dtypes=adapter_key_dtypes(adapters),
    )


def assert_same_schema(fingerprints: Sequence[Tuple[str, str]]) -> None:
    """fingerprints: list of (client_id, fingerprint)."""
    if not fingerprints:
        raise ValueError("No schema fingerprints provided")
    ref_id, ref_fp = fingerprints[0]
    for client_id, fp in fingerprints[1:]:
        if fp != ref_fp:
            raise ValueError(
                f"LoRA schema mismatch: {client_id}={fp} vs {ref_id}={ref_fp}. "
                "FedLoRA requires one common target/rank/alpha/key-shape schema."
            )


def schemas_compatible(fp_a: str, fp_b: str) -> bool:
    return fp_a == fp_b
