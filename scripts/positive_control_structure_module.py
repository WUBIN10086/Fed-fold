#!/usr/bin/env python3
"""Positive-control helpers for capacity fallback diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import torch.nn as nn

from openfold.config import model_config
from openfold.model.model import AlphaFold


def selective_unfreeze_structure_module(model: nn.Module) -> List[str]:
    """Freeze all params, then unfreeze structure_module (non-LoRA control)."""
    for param in model.parameters():
        param.requires_grad_(False)
    unfrozen = []
    module = getattr(model, "structure_module", None)
    if module is None:
        raise ValueError("model has no structure_module")
    for name, param in module.named_parameters():
        param.requires_grad_(True)
        unfrozen.append(f"structure_module.{name}")
    return unfrozen


def verify_positive_control_setup() -> dict:
    model = AlphaFold(model_config("seq_model_esm1b_ptm", train=True)).float()
    names = selective_unfreeze_structure_module(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return {
        "mode": "selective_non_lora_unfreeze",
        "scope": "structure_module",
        "unfrozen_param_tensors": len(names),
        "trainable_params": int(trainable),
        "frozen_params": int(frozen),
        "ok": trainable > 0 and frozen > 0,
        "sample_names": names[:10],
        "executed": False,
        "note": "setup-only; not a trained positive control",
    }


def should_run_real_positive_control(
    *,
    t5_capacity_v2_any_pass: bool,
    diagnostic_validation_pass: bool,
    diagnostic_validation_executed: bool,
    audit_pass: bool,
) -> bool:
    """Trigger after a complete audit and an executed failed T5 validation."""
    _ = t5_capacity_v2_any_pass  # retained in the record; both T5 routes are valid triggers
    if not audit_pass or not diagnostic_validation_executed:
        return False
    return not diagnostic_validation_pass


def positive_control_status(
    *,
    setup: dict,
    executed: bool = False,
    capacity_pass: bool = False,
    loss_decreased: Optional[bool] = None,
    tm_unchanged: Optional[bool] = None,
    tm_sparse_response: bool = False,
) -> dict:
    """Four-state positive control record for reports."""
    if not executed:
        return {
            "state": "setup_only",
            "executed": False,
            "pass": False,
            "setup": setup,
            "note": "setup-only must not classify pass/fail",
        }
    if capacity_pass:
        return {
            "state": "executed_pass",
            "executed": True,
            "pass": True,
            "setup": setup,
            "loss_decreased": loss_decreased,
            "tm_unchanged": tm_unchanged,
        }
    if loss_decreased is None:
        return {
            "state": "executed_fail_loss_trend_unknown",
            "executed": True,
            "pass": False,
            "setup": setup,
            "loss_decreased": None,
            "tm_unchanged": tm_unchanged,
        }
    if loss_decreased is False:
        return {
            "state": "executed_fail_no_loss_drop",
            "executed": True,
            "pass": False,
            "setup": setup,
            "loss_decreased": False,
            "tm_unchanged": tm_unchanged,
        }
    if tm_unchanged:
        return {
            "state": "executed_fail_loss_tm_mismatch",
            "executed": True,
            "pass": False,
            "setup": setup,
            "loss_decreased": loss_decreased,
            "tm_unchanged": True,
        }
    if tm_sparse_response:
        return {
            "state": "executed_fail_sparse_tm_response",
            "executed": True,
            "pass": False,
            "setup": setup,
            "loss_decreased": loss_decreased,
            "tm_unchanged": tm_unchanged,
            "tm_sparse_response": True,
        }
    return {
        "state": "executed_fail",
        "executed": True,
        "pass": False,
        "setup": setup,
        "loss_decreased": loss_decreased,
        "tm_unchanged": tm_unchanged,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    payload = verify_positive_control_setup()
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
