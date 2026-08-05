import json
import random
import tempfile
import unittest
from pathlib import Path

from openfold.data.data_modules import compute_sampling_audit
from scripts.build_fed_test_set import (
    difficulty_distribution,
    labels_sha256,
    split_by_cluster,
    write_reclustered_chain_cache,
)


class TestClusterSplitAndSampling(unittest.TestCase):
    def setUp(self):
        self.labels = ["A", "B", "C", "D", "E", "F"]
        self.clusters = {
            label: index for index, label in enumerate(self.labels)
        }
        self.difficulty = {
            "A": 0.4,
            "B": 0.5,
            "C": 0.7,
            "D": 0.75,
            "E": 0.9,
            "F": 0.95,
        }

    def test_fixed_seed_minimum_and_no_cluster_leakage(self):
        first = split_by_cluster(
            self.labels,
            self.clusters,
            0.2,
            random.Random(42),
            min_test_clusters=3,
            difficulty=self.difficulty,
        )
        second = split_by_cluster(
            self.labels,
            self.clusters,
            0.2,
            random.Random(42),
            min_test_clusters=3,
            difficulty=self.difficulty,
        )
        self.assertEqual(first, second)
        train, test, train_clusters, test_clusters = first
        self.assertEqual(len(test_clusters), 3)
        self.assertFalse(set(train_clusters) & set(test_clusters))
        self.assertEqual(set(train) | set(test), set(self.labels))
        distribution = difficulty_distribution(test, self.difficulty)
        self.assertEqual(
            [distribution[name] for name in ("hard", "medium", "easy")],
            [1, 1, 1],
        )

    def test_label_hash_is_order_independent(self):
        self.assertEqual(
            labels_sha256(["B", "A"]),
            labels_sha256(["A", "B"]),
        )

    def test_reclustered_cache_uses_final_train_cluster_sizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.json"
            output = Path(tmp) / "output.json"
            source.write_text(json.dumps({
                "A": {"seq": "AA", "cluster_size": 99},
                "B": {"seq": "BB", "cluster_size": 99},
                "C": {"seq": "CC", "cluster_size": 99},
            }))
            result = write_reclustered_chain_cache(
                source,
                output,
                ["A", "B", "C"],
                {"A": 1, "B": 1, "C": 2},
            )
            self.assertEqual(result["A"]["cluster_size"], 2)
            self.assertEqual(result["B"]["cluster_size"], 2)
            self.assertEqual(result["C"]["cluster_size"], 1)

    def test_sampling_audit_reports_unique_counts_and_ess(self):
        audit = compute_sampling_audit(
            ["A", "A", "B", "C"],
            {"A": "c1", "B": "c1", "C": "c2"},
        )
        self.assertEqual(audit["draw_count"], 4)
        self.assertEqual(audit["unique_label_count"], 3)
        self.assertEqual(audit["unique_cluster_count"], 2)
        self.assertAlmostEqual(audit["cluster_sampling_ess"], 1.6)
        self.assertEqual(
            audit["train_epoch_len_semantics"],
            "random_draw_count_not_dataset_pass",
        )


if __name__ == "__main__":
    unittest.main()
