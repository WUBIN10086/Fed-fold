import unittest
from pathlib import Path

from scripts.summarize_local_only import local_result_root


class TestLocalSummaryPaths(unittest.TestCase):
    def test_default_layout(self):
        self.assertEqual(
            local_result_root(Path("/run"), "client_2", "uniform", 42),
            Path("/run/clients/client_2/private/local_only/uniform/seed_42"),
        )

    def test_target_slug_layout(self):
        self.assertEqual(
            local_result_root(
                Path("/run"),
                "client_2",
                "uniform",
                42,
                target_slug="T4",
            ),
            Path("/run/clients/client_2/private/local_only/T4/uniform/seed_42"),
        )

    def test_namespaced_target_layout(self):
        self.assertEqual(
            local_result_root(
                Path("/run"),
                "client_2",
                "uniform",
                42,
                target_slug="T4",
                local_output_namespace="frozen_soft_tm_v1",
            ),
            Path(
                "/run/clients/client_2/private/"
                "frozen_soft_tm_v1/T4/uniform/seed_42"
            ),
        )


if __name__ == "__main__":
    unittest.main()
