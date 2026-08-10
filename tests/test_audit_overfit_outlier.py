#!/usr/bin/env python3
"""Tests for 9u78 audit integrity helpers (no GPU)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.audit_overfit_outlier import (
    assert_unique_prediction_dirs,
    build_audit_summary,
    parse_pdb_chain,
)


class TestAuditOutlier(unittest.TestCase):
    def test_duplicate_prediction_dirs_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            d = root / "preds"
            d.mkdir()
            errs = assert_unique_prediction_dirs([d, d])
            self.assertIn("duplicate_prediction_directories", errs)

    def test_build_audit_summary_fail_closed_on_mapping_error(self):
        static = {
            "label": "9u78_A",
            "mapping_errors": ["native_chain_mismatch"],
            "audit_integrity_pass": False,
        }
        rescore = {"metric_reproducible": True, "tm_selected_mean": 0.937}
        summary = build_audit_summary(static, rescore, None)
        self.assertFalse(summary["audit_integrity_pass"])
        self.assertTrue(summary["metric_reproducible"])
        self.assertFalse(summary["audit_complete"])
        self.assertFalse(summary["audit_pass"])

    def test_independent_replay_is_required_and_integrity_checked(self):
        static = {
            "label": "9u78_A",
            "mapping_errors": [],
            "audit_integrity_pass": True,
        }
        rescore = {"metric_reproducible": True, "tm_selected_mean": 0.937}
        incomplete = build_audit_summary(static, rescore, None)
        self.assertFalse(incomplete["audit_complete"])
        self.assertIsNone(incomplete["inference_reproducible"])

        replay = {
            "inference_reproducible": True,
            "cache_reuse_detected": False,
            "integrity_errors": [],
            "tm_values": [0.937, 0.937],
            "lddt_values": [0.8, 0.8],
            "replay_pairwise_ca_rmsd": 0.0,
        }
        complete = build_audit_summary(static, rescore, replay)
        self.assertTrue(complete["audit_complete"])
        self.assertTrue(complete["audit_pass"])

        replay["integrity_errors"] = ["replay_0_output_sha256_mismatch"]
        failed = build_audit_summary(static, rescore, replay)
        self.assertFalse(failed["audit_integrity_pass"])
        self.assertFalse(failed["audit_pass"])

    def test_parse_minimal_pdb(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdb = Path(tmp) / "x.pdb"
            pdb.write_text(
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
                "ATOM      2  CA  GLY A   2       1.000   0.000   0.000  1.00  0.00           C\n"
            )
            info = parse_pdb_chain(pdb)
            self.assertEqual(info["n_ca"], 2)
            self.assertEqual(info["sequence"], "AG")


if __name__ == "__main__":
    unittest.main()
