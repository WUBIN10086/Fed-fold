#!/usr/bin/env python3
"""Tests for fail-closed target-ablation artifact reuse."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_client0_target_ablation import (
    POSITIVE_CONTROL_CHECKPOINT_WEIGHTS_SOURCE,
    assert_manifest_compatible,
    lora_export_is_compatible,
    prepare_run_manifest,
    sha256_file,
)


class TestTargetAblationManifests(unittest.TestCase):
    def test_positive_control_uses_raw_converted_state_dict(self):
        self.assertEqual(POSITIVE_CONTROL_CHECKPOINT_WEIGHTS_SOURCE, "state_dict")

    def test_matching_manifest_reuses_and_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "local_run.json"
            expected = {"target": "input_embedder", "seed": 42}
            path.write_text(json.dumps(expected))
            assert_manifest_compatible(path, expected, "unit")
            with self.assertRaisesRegex(RuntimeError, "new namespace"):
                assert_manifest_compatible(
                    path, {**expected, "seed": 43}, "unit"
                )

    def test_artifact_without_manifest_is_never_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "local_run.json"
            with self.assertRaisesRegex(RuntimeError, "without local_run.json"):
                prepare_run_manifest(
                    path, {"seed": 42}, artifact_exists=True, context="unit"
                )

    def test_export_binds_parent_checkpoint_and_output_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "adapter.ckpt"
            parent = root / "parent.pt"
            model = root / "model.pt"
            checkpoint.write_bytes(b"adapter")
            parent.write_bytes(b"parent")
            model.write_bytes(b"model")
            fingerprint = "f" * 64
            manifest = {
                "output": str(model),
                "lora_rank": 4,
                "lora_alpha": 8.0,
                "lora_scale": 1.0,
                "lora_target": "input_embedder",
                "base_checkpoint_sha256": sha256_file(parent),
                "adapter_checkpoint_sha256": sha256_file(checkpoint),
                "output_sha256": sha256_file(model),
                "scale_semantics": "unit_raw_delta",
                "lora_schema_fingerprint": fingerprint,
                "adapter_key_count": 10,
                "adapter_shapes": {"x.lora_A": [4, 8]},
                "adapter_dtypes": {"x.lora_A": "float32"},
                "report": {"lora_schema_fingerprint": fingerprint},
            }
            model.with_suffix(".export_manifest.json").write_text(
                json.dumps(manifest)
            )
            compatible, errors = lora_export_is_compatible(
                checkpoint=checkpoint,
                model=model,
                parent=parent,
                target="input_embedder",
                scale=1.0,
            )
            self.assertTrue(compatible, errors)
            checkpoint.write_bytes(b"changed")
            compatible, errors = lora_export_is_compatible(
                checkpoint=checkpoint,
                model=model,
                parent=parent,
                target="input_embedder",
                scale=1.0,
            )
            self.assertFalse(compatible)
            self.assertIn("adapter_checkpoint_sha256_mismatch", errors)


if __name__ == "__main__":
    unittest.main()
