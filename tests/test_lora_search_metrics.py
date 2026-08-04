import unittest

from scripts.lora_search_metrics import (
    cluster_macro_metrics,
    scale_grid,
    select_model_with_baseline,
)
from scripts.run_client1_lora_search import (
    export_command,
    inference_command,
    train_command,
)


class TestLoRASearchMetrics(unittest.TestCase):
    def test_cluster_macro_differs_from_sample_mean(self):
        metrics = cluster_macro_metrics(
            {"A": 1.0, "B": 1.0, "C": 0.0},
            {"A": "large", "B": "large", "C": "small"},
        )
        self.assertAlmostEqual(metrics["sample_mean_tm"], 2 / 3)
        self.assertAlmostEqual(metrics["cluster_macro_tm"], 0.5)
        self.assertEqual(metrics["cluster_count"], 2)

    def test_cluster_deltas_and_worst_cluster(self):
        metrics = cluster_macro_metrics(
            {"A": 0.9, "B": 0.6},
            {"A": "c1", "B": "c2"},
            {"A": 0.8, "B": 0.7},
        )
        self.assertAlmostEqual(metrics["cluster_delta"]["c1"], 0.1)
        self.assertAlmostEqual(metrics["cluster_delta"]["c2"], -0.1)
        self.assertAlmostEqual(metrics["worst_cluster_delta"], -0.1)

    def test_all_lora_below_baseline_selects_baseline(self):
        baseline = {
            "candidate_id": "baseline",
            "cluster_macro_tm": 0.8,
            "sample_mean_tm": 0.79,
        }
        selected = select_model_with_baseline(
            [
                {"candidate_id": "a", "cluster_macro_tm": 0.799},
                {"candidate_id": "b", "cluster_macro_tm": 0.7},
            ],
            baseline,
        )
        self.assertEqual(selected["selected_model"], "baseline")

    def test_min_delta_and_best_eligible(self):
        baseline = {"candidate_id": "baseline", "cluster_macro_tm": 0.8}
        selected = select_model_with_baseline(
            [
                {"candidate_id": "a", "cluster_macro_tm": 0.805},
                {"candidate_id": "b", "cluster_macro_tm": 0.82},
            ],
            baseline,
            min_delta=0.01,
        )
        self.assertEqual(selected["selected_model"], "b")

    def test_scale_grid_includes_noop(self):
        self.assertEqual(scale_grid(), (0.0, 0.1, 0.25, 0.5, 0.75, 1.0))

    def test_v2_commands_pin_weight_sources_seed_and_scale(self):
        row = {
            "rank": "4",
            "alpha": "8",
            "dropout": "0",
            "target": "structure_module.ipa",
        }
        train = [str(value) for value in train_command(
            row,
            "out",
            "labels",
            10,
            1,
        )]
        self.assertEqual(
            train[train.index("--init_weights_source") + 1],
            "ema",
        )
        export = [str(value) for value in export_command("in", "out", 0.25)]
        self.assertEqual(
            export[export.index("--base-weights-source") + 1],
            "ema",
        )
        self.assertEqual(
            export[export.index("--adapter-weights-source") + 1],
            "model",
        )
        self.assertEqual(export[export.index("--lora-scale") + 1], "0.25")
        inference = [str(value) for value in inference_command(
            "fasta",
            "cif",
            "alignments",
            "predictions",
            "checkpoint",
            "auto",
        )]
        self.assertEqual(
            inference[inference.index("--data_random_seed") + 1],
            "42",
        )


if __name__ == "__main__":
    unittest.main()
