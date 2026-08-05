import unittest

import torch

from openfold.utils.lr_schedulers import (
    AlphaFoldLRScheduler,
    compute_alphafold_learning_rate,
)


class TestAlphaFoldLRScheduler(unittest.TestCase):
    def test_expected_warmup_plateau_and_decay(self):
        kwargs = {
            "base_lr": 0.0,
            "max_lr": 1e-4,
            "warmup_no_steps": 20,
            "start_decay_after_n_steps": 100,
            "decay_every_n_steps": 10,
            "decay_factor": 0.5,
        }
        self.assertEqual(compute_alphafold_learning_rate(0, **kwargs), 0.0)
        self.assertAlmostEqual(
            compute_alphafold_learning_rate(10, **kwargs),
            5e-5,
        )
        self.assertAlmostEqual(
            compute_alphafold_learning_rate(20, **kwargs),
            1e-4,
        )
        self.assertAlmostEqual(
            compute_alphafold_learning_rate(21, **kwargs),
            1e-4,
        )
        self.assertAlmostEqual(
            compute_alphafold_learning_rate(101, **kwargs),
            5e-5,
        )

    def test_zero_warmup_starts_at_max_lr(self):
        self.assertAlmostEqual(
            compute_alphafold_learning_rate(
                0,
                max_lr=2e-4,
                warmup_no_steps=0,
                start_decay_after_n_steps=10,
            ),
            2e-4,
        )

    def test_scheduler_uses_configured_values(self):
        parameter = torch.nn.Parameter(torch.zeros(()))
        optimizer = torch.optim.Adam([parameter], lr=1.0)
        scheduler = AlphaFoldLRScheduler(
            optimizer,
            base_lr=0.0,
            max_lr=1e-4,
            warmup_no_steps=2,
            start_decay_after_n_steps=5,
            decay_every_n_steps=2,
            decay_factor=0.5,
        )
        self.assertAlmostEqual(scheduler.get_last_lr()[0], 0.0)
        optimizer.step()
        scheduler.step()
        self.assertAlmostEqual(scheduler.get_last_lr()[0], 5e-5)
        optimizer.step()
        scheduler.step()
        self.assertAlmostEqual(scheduler.get_last_lr()[0], 1e-4)

    def test_resume_step(self):
        parameter = torch.nn.Parameter(torch.zeros(()))
        optimizer = torch.optim.Adam([parameter], lr=1e-4)
        optimizer.param_groups[0]["initial_lr"] = 1e-4
        scheduler = AlphaFoldLRScheduler(
            optimizer,
            last_epoch=20,
            max_lr=1e-4,
            warmup_no_steps=20,
            start_decay_after_n_steps=100,
        )
        self.assertEqual(scheduler.last_epoch, 21)
        self.assertAlmostEqual(scheduler.get_last_lr()[0], 1e-4)

    def test_invalid_decay_settings(self):
        with self.assertRaises(ValueError):
            compute_alphafold_learning_rate(
                1,
                decay_every_n_steps=0,
            )


if __name__ == "__main__":
    unittest.main()
