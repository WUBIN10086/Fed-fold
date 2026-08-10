import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.select_local_checkpoint import score_candidate, select_candidate


FIELDS = [
    "label",
    "client",
    "cluster_id",
    "difficulty",
    "tm_baseline",
    "tm_model",
    "delta_tm",
    "lddt_baseline",
    "lddt_model",
    "delta_lddt_ca",
]


class TestLocalCheckpointSelection(unittest.TestCase):
    def write_candidate(
        self,
        root: Path,
        name: str,
        epoch: int,
        scale: float,
        hard_delta: float,
        nonhard_delta: float,
    ) -> Path:
        candidate = root / name
        candidate.mkdir()
        paired = candidate / "paired_deltas.csv"
        with paired.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerow(
                {
                    "label": "H",
                    "client": "client_0",
                    "cluster_id": "c1",
                    "difficulty": "hard",
                    "tm_baseline": 0.3,
                    "tm_model": 0.3 + hard_delta,
                    "delta_tm": hard_delta,
                    "delta_lddt_ca": "",
                }
            )
            writer.writerow(
                {
                    "label": "E",
                    "client": "client_0",
                    "cluster_id": "c2",
                    "difficulty": "easy",
                    "tm_baseline": 0.9,
                    "tm_model": 0.9 + nonhard_delta,
                    "delta_tm": nonhard_delta,
                    "delta_lddt_ca": "",
                }
            )
        (candidate / "candidate.json").write_text(
            json.dumps(
                {
                    "epoch": epoch,
                    "scale": scale,
                    "model_path": str(candidate / "model.pt"),
                }
            ),
            encoding="utf-8",
        )
        return paired

    def test_selects_best_hard_gain_under_constraint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = self.write_candidate(root, "good", 2, 0.5, 0.04, -0.001)
            unsafe = self.write_candidate(root, "unsafe", 3, 1.0, 0.10, -0.02)
            decision = select_candidate(
                [score_candidate(good), score_candidate(unsafe)]
            )
            self.assertFalse(decision["fallback_to_baseline"])
            self.assertEqual(decision["selected_epoch"], 2)
            self.assertEqual(decision["selected_scale"], 0.5)

    def test_falls_back_for_nonpositive_hard_gain(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = self.write_candidate(
                root, "candidate", 1, 1.0, -0.01, 0.0
            )
            decision = select_candidate([score_candidate(candidate)])
            self.assertTrue(decision["fallback_to_baseline"])
            self.assertEqual(decision["selected_scale"], 0.0)


if __name__ == "__main__":
    unittest.main()
