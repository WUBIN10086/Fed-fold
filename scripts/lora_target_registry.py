#!/usr/bin/env python3
"""Canonical LoRA target matrix for client0 ablation experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional


@dataclass(frozen=True)
class TargetSpec:
    slug: str
    target: str
    expected_modules: int
    expected_trainable: int
    role: str
    description: str


# Counts verified on seq_model_esm1b_ptm with rank=4.
TARGET_SPECS: Dict[str, TargetSpec] = {
    "T0": TargetSpec(
        slug="T0",
        target=(
            "structure_module.ipa,"
            "structure_module.transition,"
            "structure_module.bb_update"
        ),
        expected_modules=10,
        expected_trainable=32072,
        role="primary",
        description="Current core structure module",
    ),
    "T1": TargetSpec(
        slug="T1",
        target="structure_module",
        expected_modules=18,
        expected_trainable=43904,
        role="primary",
        description="Full structure module",
    ),
    "T2": TargetSpec(
        slug="T2",
        target="structure_module,evoformer.linear",
        expected_modules=19,
        expected_trainable=46464,
        role="primary",
        description="Structure plus Evoformer output projection",
    ),
    "T3": TargetSpec(
        slug="T3",
        target="structure_module,evoformer.linear,evoformer.blocks.47",
        expected_modules=56,
        expected_trainable=103104,
        role="primary",
        description="Structure plus last Evoformer block",
    ),
    "T4": TargetSpec(
        slug="T4",
        target=(
            "structure_module,evoformer.linear,"
            "evoformer.blocks.44,evoformer.blocks.45,"
            "evoformer.blocks.46,evoformer.blocks.47"
        ),
        expected_modules=167,
        expected_trainable=273024,
        role="primary",
        description="Structure plus last four Evoformer blocks",
    ),
    "T5A": TargetSpec(
        slug="T5A",
        target="input_embedder",
        expected_modules=5,
        expected_trainable=19292,
        role="fallback",
        description="Input embedder branch only",
    ),
    "T5B": TargetSpec(
        slug="T5B",
        target="structure_module,input_embedder",
        expected_modules=23,
        expected_trainable=63196,
        role="fallback",
        description="Structure plus input embedder",
    ),
}

PRIMARY_OVERFIT_TARGETS = ("T0", "T1", "T3", "T4")
PRIMARY_VALIDATION_TARGETS = ("T0", "T1", "T2", "T3", "T4")
FALLBACK_TARGETS = ("T5A", "T5B")
FALLBACK_VALIDATION_TARGETS = ("T5A", "T5B")

# Deterministic 8 hardest train clusters for client0 overfit check.
OVERFIT8_LABELS = (
    "9fei_A",
    "9wye_A",
    "9b3e_A",
    "9ftf_A",
    "9i2o_A",
    "9t2d_A",
    "9xpd_A",
    "9u78_A",
)
OVERFIT8_CLUSTERS = (
    "c50",
    "c772",
    "c207",
    "c343",
    "c496",
    "c167",
    "c752",
    "c935",
)

CAPACITY_CHECKPOINTS = (40, 80, 120, 200)
OVERFIT_EPOCH_LEN = 8
OVERFIT_MAX_EPOCHS = 25  # 25 * 8 = 200 steps
OVERFIT_WARMUP_STEPS = 5


def get_target(slug: str) -> TargetSpec:
    if slug not in TARGET_SPECS:
        raise KeyError(f"Unknown target slug: {slug}")
    return TARGET_SPECS[slug]


def target_dict(slug: Optional[str] = None) -> Mapping[str, object]:
    if slug is None:
        return {
            key: {
                "slug": spec.slug,
                "target": spec.target,
                "expected_modules": spec.expected_modules,
                "expected_trainable": spec.expected_trainable,
                "role": spec.role,
                "description": spec.description,
            }
            for key, spec in TARGET_SPECS.items()
        }
    spec = get_target(slug)
    return {
        "slug": spec.slug,
        "target": spec.target,
        "expected_modules": spec.expected_modules,
        "expected_trainable": spec.expected_trainable,
        "role": spec.role,
        "description": spec.description,
    }


def list_slugs(role: Optional[str] = None) -> List[str]:
    if role is None:
        return list(TARGET_SPECS)
    return [slug for slug, spec in TARGET_SPECS.items() if spec.role == role]
