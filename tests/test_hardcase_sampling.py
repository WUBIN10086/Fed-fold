"""Tests for hard-aware sampling helpers and difficulty bands."""

from __future__ import annotations

import unittest

from openfold.data.data_modules import (
    allocate_stratum_counts,
    compute_sampling_audit,
    parse_hard_aware_ratios,
)
from scripts.build_fed_test_set import difficulty_band, difficulty_weight


class TestHardcaseSampling(unittest.TestCase):
    def test_difficulty_band_threshold_0_5(self):
        self.assertEqual(difficulty_band(0.499), "hard")
        self.assertEqual(difficulty_band(0.5), "medium")
        self.assertEqual(difficulty_band(0.799), "medium")
        self.assertEqual(difficulty_band(0.8), "easy")
        self.assertEqual(difficulty_band(None), "unknown")

    def test_difficulty_weight(self):
        self.assertAlmostEqual(difficulty_weight(0.0), 1.0)
        self.assertAlmostEqual(difficulty_weight(0.25), 0.5)
        self.assertAlmostEqual(difficulty_weight(0.5), 0.0)
        self.assertAlmostEqual(difficulty_weight(0.9), 0.0)

    def test_ratio_parse_and_fallback(self):
        ratios = parse_hard_aware_ratios("0.7,0.15,0.15")
        self.assertAlmostEqual(sum(ratios.values()), 1.0)
        counts, fallback, reason = allocate_stratum_counts(
            10,
            ratios,
            {"hard": 5, "medium": 0, "easy": 5},
        )
        self.assertTrue(fallback)
        self.assertEqual(reason, None)
        self.assertEqual(counts["medium"], 0)
        self.assertEqual(sum(counts.values()), 10)

    def test_no_hard_fallback(self):
        counts, fallback, reason = allocate_stratum_counts(
            8,
            parse_hard_aware_ratios("0.7,0.15,0.15"),
            {"hard": 0, "medium": 3, "easy": 3},
        )
        self.assertTrue(fallback)
        self.assertEqual(reason, "no_hard_local_data")
        self.assertEqual(sum(counts.values()), 0)

    def test_sampling_audit_difficulty_fields(self):
        audit = compute_sampling_audit(
            ["A", "A", "B", "C"],
            {"A": "c1", "B": "c1", "C": "c2"},
            label_to_difficulty={"A": "hard", "B": "medium", "C": "easy"},
            target_ratios={"hard": 0.7, "medium": 0.15, "easy": 0.15},
            fallback_applied=False,
            cumulative_labels={"A", "B", "C", "D"},
        )
        self.assertEqual(audit["hard_draws"], 2)
        self.assertEqual(audit["medium_draws"], 1)
        self.assertEqual(audit["easy_draws"], 1)
        self.assertEqual(audit["unique_hard_labels"], 1)
        self.assertEqual(audit["cumulative_label_coverage"], 4)


if __name__ == "__main__":
    unittest.main()
