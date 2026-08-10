#!/usr/bin/env python3
"""Tests for target ablation path isolation."""

from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SHELL = REPO / "scripts" / "fed_lora_hardcase_fed.sh"


class TestTargetAblationPaths(unittest.TestCase):
    def _local_only_root(self, env: dict) -> str:
        script = r"""
source "$1"
local_only_root 0 uniform 42
"""
        merged = os.environ.copy()
        merged.update(env)
        merged["RUN_ROOT"] = str(REPO / "outputs" / "fed_lora_hardcase_fed_v1")
        proc = subprocess.run(
            ["bash", "-c", script, "_", str(SHELL)],
            check=True,
            capture_output=True,
            text=True,
            env=merged,
            cwd=str(REPO),
        )
        return proc.stdout.strip().splitlines()[-1]

    def test_default_local_only_path_unchanged(self):
        path = self._local_only_root(
            {
                "LOCAL_OUTPUT_NAMESPACE": "",
                "TARGET_SLUG": "",
            }
        )
        self.assertTrue(path.endswith("private/local_only/uniform/seed_42"), path)

    def test_target_namespace_isolation(self):
        paths = []
        for slug in ("T0", "T1", "T2", "T3", "T4"):
            path = self._local_only_root(
                {
                    "LOCAL_OUTPUT_NAMESPACE": "target_ablation_v1",
                    "TARGET_SLUG": slug,
                }
            )
            paths.append(path)
            self.assertIn(f"target_ablation_v1/{slug}/uniform/seed_42", path)
            self.assertNotIn("/local_only/", path)
        self.assertEqual(len(set(paths)), 5)


if __name__ == "__main__":
    unittest.main()
