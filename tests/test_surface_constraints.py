import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sample_augmentation/scripts"))
from surface_constraints import surface_losses


class SurfaceConstraintTests(unittest.TestCase):
    def test_rejects_split_surfaces_with_correct_mean_depth(self):
        edges = torch.tensor([[0.5, 1.5, 2.5, 3.5]])
        ranges = torch.tensor([2.0])
        valid = torch.tensor([True])
        correct = surface_losses(torch.tensor([[0.0, 1.0, 0.0]]), edges, ranges, valid)
        split = surface_losses(torch.tensor([[0.5, 0.0, 0.5]]), edges, ranges, valid)
        self.assertAlmostEqual(float(correct["depth"]), float(split["depth"]))
        self.assertGreater(float(split["surface"]), float(correct["surface"]) + 1)
        self.assertGreater(float(split["free_space"]), float(correct["free_space"]))

    def test_unreliable_and_nonfinite_depths_do_not_supervise(self):
        weights = torch.tensor([[0.3, 0.4], [0.2, 0.6]], requires_grad=True)
        losses = surface_losses(weights, torch.tensor([[0.1, 1.0, 2.0]]).expand(2, -1),
                                torch.tensor([float("nan"), 1.2]), torch.tensor([True, False]))
        sum(losses.values()).backward()
        self.assertEqual(float(weights.grad.abs().sum()), 0)
        self.assertTrue(all(float(value) == 0 for value in losses.values()))

    def test_gradients_remove_near_occluder_and_add_surface_mass(self):
        weights = torch.tensor([[0.5, 0.4, 0.05]], requires_grad=True)
        losses = surface_losses(weights, torch.tensor([[0.5, 1.5, 2.5, 3.5]]),
                                torch.tensor([2.0]), torch.tensor([True]))
        (losses["surface"] + losses["free_space"]).backward()
        self.assertGreater(float(weights.grad[0, 0]), 0)
        self.assertLess(float(weights.grad[0, 1]), 0)


if __name__ == "__main__":
    unittest.main()
