"""Regression tests for checkpointed trainable modules with frozen inputs."""

from __future__ import annotations

import unittest

import torch

from openfold.utils.checkpointing import checkpoint_blocks


class _Scale(torch.nn.Module):
    def __init__(self, value: float, trainable: bool):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.tensor(value), requires_grad=trainable
        )

    def forward(self, x):
        return x * self.weight


class TestCheckpointedLoRAGrad(unittest.TestCase):
    def test_trainable_parameter_gets_gradient_from_frozen_input(self):
        frozen = _Scale(3.0, trainable=False)
        adapter = _Scale(2.0, trainable=True)
        x = torch.tensor(4.0, requires_grad=False)

        output, = checkpoint_blocks(
            [frozen, adapter], args=(x,), blocks_per_ckpt=1
        )
        output.backward()

        self.assertIsNotNone(adapter.weight.grad)
        torch.testing.assert_close(adapter.weight.grad, torch.tensor(12.0))
        self.assertIsNone(frozen.weight.grad)
        self.assertIsNone(x.grad)


if __name__ == "__main__":
    unittest.main()
