import tempfile
import unittest
from pathlib import Path

from scripts.generate_client1_lora_v2_matrix import generate


class TestLoRAV2Matrix(unittest.TestCase):
    def test_matrix_contains_baseline_and_full_grid_without_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, commands, rows = generate(Path(tmp))
            self.assertTrue(manifest.exists())
            self.assertTrue(commands.exists())
            self.assertEqual(len(rows), 271)
            self.assertEqual(rows[0]["candidate_type"], "baseline")
            lora = rows[1:]
            self.assertEqual(
                {row["learning_rate"] for row in lora},
                {3e-6, 1e-5, 3e-5},
            )
            self.assertEqual(
                {row["step"] for row in lora},
                {5, 10, 20, 40, 80},
            )
            self.assertEqual({row["seed"] for row in lora}, {42, 43, 44})
            self.assertEqual(
                {row["lora_scale"] for row in lora},
                {0, 0.1, 0.25, 0.5, 0.75, 1},
            )
            command_text = commands.read_text()
            self.assertEqual(command_text.count("test -d "), 45)
            self.assertIn("--init_weights_source ema", command_text)


if __name__ == "__main__":
    unittest.main()
