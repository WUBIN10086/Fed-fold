import json
import random
import tempfile
import unittest
from pathlib import Path

import torch

from openfold.data.data_modules import (
    OpenFoldDataset,
    add_baseline_preservation_features,
    align_baseline_ca_to_sequence,
    compute_sampling_audit,
    difficulty_conditioned_fape_clamp_value,
    read_pdb_ca_coordinates,
)
from openfold.utils.loss import (
    baseline_distance_preservation_loss,
    baseline_guided_hard_correction_loss,
    baseline_guided_hard_soft_tm_loss,
)
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

    def test_difficulty_conditioned_fape_clamp_value(self):
        self.assertEqual(
            difficulty_conditioned_fape_clamp_value("hard"), 0.0
        )
        self.assertEqual(
            difficulty_conditioned_fape_clamp_value("medium"), 1.0
        )
        self.assertEqual(
            difficulty_conditioned_fape_clamp_value("easy"), 1.0
        )
        self.assertIsNone(
            difficulty_conditioned_fape_clamp_value("unknown")
        )

    def test_dataset_overrides_clamp_by_difficulty(self):
        import torch

        class FakeDataset:
            chain_data_cache = {
                "H": {"seq": "AA"},
                "E": {"seq": "AA"},
            }

            def __len__(self):
                return 2

            def idx_to_chain_id(self, idx):
                return ("H", "E")[idx]

            def __getitem__(self, idx):
                return {
                    "use_clamped_fape": torch.zeros(4, dtype=torch.float32)
                }

        dataset = OpenFoldDataset(
            datasets=[FakeDataset()],
            probabilities=[1.0],
            epoch_len=2,
            _roll_at_init=False,
            difficulty_by_chain={"H": "hard", "E": "easy"},
            difficulty_conditioned_fape=True,
        )
        dataset.datapoints = [(0, 0), (0, 1)]
        self.assertTrue(torch.equal(
            dataset[0]["use_clamped_fape"], torch.zeros(4)
        ))
        self.assertTrue(torch.equal(
            dataset[1]["use_clamped_fape"], torch.ones(4)
        ))

    def test_baseline_preservation_features_follow_crop_indices(self):
        features = {
            "residue_index": torch.tensor([
                [0, 0],
                [2, 2],
                [0, 0],
            ]),
            "seq_mask": torch.tensor([
                [1.0, 1.0],
                [1.0, 1.0],
                [0.0, 0.0],
            ]),
        }
        baseline_ca = torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ])
        output = add_baseline_preservation_features(
            features, baseline_ca, active=True
        )
        self.assertEqual(
            output["baseline_ca_positions"].shape,
            torch.Size([3, 3, 2]),
        )
        self.assertTrue(torch.equal(
            output["baseline_ca_positions"][1, :, 0],
            baseline_ca[2],
        ))
        self.assertTrue(torch.equal(
            output["baseline_ca_mask"][:, 0],
            torch.tensor([1.0, 1.0, 0.0]),
        ))
        self.assertTrue(torch.equal(
            output["baseline_preservation_active"],
            torch.ones(2),
        ))
        self.assertTrue(torch.equal(
            output["baseline_hard_active"],
            torch.zeros(2),
        ))

        hard_output = add_baseline_preservation_features(
            features, baseline_ca, active=False, hard_active=True
        )
        self.assertTrue(torch.equal(
            hard_output["baseline_hard_active"],
            torch.ones(2),
        ))

    def test_align_baseline_ca_masks_missing_residue(self):
        baseline_ca = torch.tensor([
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ])
        aligned, mask = align_baseline_ca_to_sequence(
            "AC", baseline_ca, "ABC"
        )
        self.assertTrue(torch.equal(
            mask, torch.tensor([1.0, 0.0, 1.0])
        ))
        self.assertTrue(torch.equal(aligned[0], baseline_ca[0]))
        self.assertTrue(torch.equal(aligned[2], baseline_ca[1]))

    def test_read_pdb_ca_coordinates(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdb = Path(tmp) / "baseline.pdb"
            pdb.write_text(
                "ATOM      1  CA  ALA A   1       1.000   2.000   3.000"
                "  1.00  0.00           C\n"
                "ATOM      2  CA  GLY A   2       4.000   5.000   6.000"
                "  1.00  0.00           C\n"
                "END\n"
            )
            coords = read_pdb_ca_coordinates(str(pdb))
        self.assertTrue(torch.equal(
            coords,
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        ))

    def test_baseline_distance_preservation_respects_active_examples(self):
        atom_count = 37
        pred = torch.zeros((2, 3, atom_count, 3), dtype=torch.float32)
        target = torch.tensor([
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        ])
        ca_index = 1
        pred[0, :, ca_index, :] = target[0]
        pred[1, :, ca_index, 0] = torch.tensor([0.0, 2.0, 4.0])
        mask = torch.ones((2, 3), dtype=torch.float32)

        first_only = baseline_distance_preservation_loss(
            pred, target, mask, torch.tensor([1.0, 0.0])
        )
        second_only = baseline_distance_preservation_loss(
            pred, target, mask, torch.tensor([0.0, 1.0])
        )
        self.assertAlmostEqual(float(first_only), 0.0, places=7)
        self.assertGreater(float(second_only), 0.0)


    def test_hard_correction_rewards_margin_improvement(self):
        atom_count = 37
        native_ca = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ])
        baseline_ca = native_ca.clone()
        baseline_ca[-1, 0] = 6.0
        native = torch.zeros((1, 4, atom_count, 3))
        baseline_pred = torch.zeros((1, 4, atom_count, 3))
        corrected_pred = torch.zeros((1, 4, atom_count, 3))
        native[0, :, 1, :] = native_ca
        baseline_pred[0, :, 1, :] = baseline_ca
        corrected_pred[0, :, 1, :] = native_ca
        atom_mask = torch.ones((1, 4, atom_count))
        baseline_mask = torch.ones((1, 4))

        baseline_loss = baseline_guided_hard_correction_loss(
            baseline_pred,
            native,
            atom_mask,
            baseline_ca.unsqueeze(0),
            baseline_mask,
            torch.ones(1),
            margin=0.5,
            min_baseline_error=0.5,
            min_sequence_separation=1,
        )
        corrected_loss = baseline_guided_hard_correction_loss(
            corrected_pred,
            native,
            atom_mask,
            baseline_ca.unsqueeze(0),
            baseline_mask,
            torch.ones(1),
            margin=0.5,
            min_baseline_error=0.5,
            min_sequence_separation=1,
        )
        self.assertGreater(float(baseline_loss), 0.49)
        self.assertAlmostEqual(float(corrected_loss), 0.0, places=7)

    def test_hard_correction_respects_inactive_examples(self):
        atom_count = 37
        native = torch.zeros((1, 4, atom_count, 3))
        prediction = torch.zeros((1, 4, atom_count, 3))
        baseline_ca = torch.zeros((1, 4, 3))
        native[0, :, 1, 0] = torch.arange(4, dtype=torch.float32)
        prediction[0, :, 1, 0] = torch.tensor([0.0, 1.0, 2.0, 6.0])
        baseline_ca[0, :, 0] = torch.tensor([0.0, 1.0, 2.0, 6.0])
        loss = baseline_guided_hard_correction_loss(
            prediction,
            native,
            torch.ones((1, 4, atom_count)),
            baseline_ca,
            torch.ones((1, 4)),
            torch.zeros(1),
            margin=0.5,
            min_baseline_error=0.5,
            min_sequence_separation=1,
        )
        self.assertAlmostEqual(float(loss), 0.0, places=7)


    def test_hard_soft_tm_is_rigid_invariant_and_has_gradient(self):
        atom_count = 37
        native_ca = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [2.0, 2.0, 1.0],
            [3.0, 1.0, 4.0],
        ])
        baseline_ca = native_ca.clone()
        baseline_ca[-2:] += torch.tensor([5.0, -4.0, 3.0])
        rotation = torch.tensor([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        rigid_ca = native_ca @ rotation + torch.tensor([7.0, -3.0, 2.0])
        native = torch.zeros((1, 6, atom_count, 3))
        baseline_pred = torch.zeros((1, 6, atom_count, 3))
        rigid_pred = torch.zeros((1, 6, atom_count, 3))
        native[0, :, 1, :] = native_ca
        baseline_pred[0, :, 1, :] = baseline_ca
        rigid_pred[0, :, 1, :] = rigid_ca
        atom_mask = torch.ones((1, 6, atom_count))
        baseline_mask = torch.ones((1, 6))

        baseline_pred.requires_grad_()
        baseline_loss = baseline_guided_hard_soft_tm_loss(
            baseline_pred,
            native,
            atom_mask,
            baseline_ca.unsqueeze(0),
            baseline_mask,
            torch.ones(1),
            score_margin=0.02,
        )
        rigid_loss = baseline_guided_hard_soft_tm_loss(
            rigid_pred,
            native,
            atom_mask,
            baseline_ca.unsqueeze(0),
            baseline_mask,
            torch.ones(1),
            score_margin=0.02,
        )
        baseline_loss.backward()
        self.assertGreater(float(baseline_loss), 0.019)
        self.assertAlmostEqual(float(rigid_loss), 0.0, places=6)
        self.assertTrue(torch.isfinite(baseline_pred.grad).all())
        self.assertGreater(
            float(baseline_pred.grad[0, :, 1, :].abs().sum()),
            0.0,
        )

    def test_hard_soft_tm_respects_inactive_examples(self):
        atom_count = 37
        native = torch.zeros((1, 4, atom_count, 3))
        prediction = torch.zeros((1, 4, atom_count, 3))
        baseline = torch.zeros((1, 4, 3))
        native[0, :, 1, 0] = torch.arange(4, dtype=torch.float32)
        prediction[0, :, 1, 0] = torch.tensor([0.0, 1.0, 2.0, 6.0])
        baseline[0, :, 0] = torch.tensor([0.0, 1.0, 2.0, 6.0])
        loss = baseline_guided_hard_soft_tm_loss(
            prediction,
            native,
            torch.ones((1, 4, atom_count)),
            baseline,
            torch.ones((1, 4)),
            torch.zeros(1),
        )
        self.assertAlmostEqual(float(loss), 0.0, places=7)


if __name__ == "__main__":
    unittest.main()
