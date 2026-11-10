import sys
from pathlib import Path
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sample_augmentation/scripts"))
from nerf_diagnostic_data import camera_rays, llff_to_opengl, reference_order
from run_nerf_diagnostic import composite, sample_pdf


class NeRFDiagnosticTests(unittest.TestCase):
    def test_ray_normalization_preserves_z_depth(self):
        pose = torch.eye(4)
        intrinsics = torch.tensor([[3.0, 0, 2.0], [0, 3.0, 1.5], [0, 0, 1.0]])
        origins, directions, lengths = camera_rays(pose, intrinsics, 3, 4)
        points = origins + directions * (2 * lengths[:, None])
        torch.testing.assert_close(points[:, 2], torch.full((12,), -2.0))
        projected_x = points[:, 0] / -points[:, 2] * 3 + 2
        torch.testing.assert_close(projected_x, torch.arange(4).repeat(3).float() + 0.5)

    def test_llff_axis_conversion(self):
        raw = np.zeros((1, 3, 5))
        raw[0, :, 0] = [0, -1, 0]
        raw[0, :, 1] = [1, 0, 0]
        raw[0, :, 2] = [0, 0, 1]
        raw[0, :, 3] = [1, 2, 3]
        poses = llff_to_opengl(raw)
        np.testing.assert_allclose(poses[0, :3, :3], np.eye(3))
        np.testing.assert_allclose(poses[0, :3, 3], [1, 2, 3])

    def test_pdf_handles_empty_density_and_stays_in_bounds(self):
        edges = torch.tensor([[2.0, 3.0, 4.0, 6.0]])
        samples = sample_pdf(edges, torch.zeros(1, 3), 32, False)
        self.assertTrue(torch.isfinite(samples).all())
        self.assertGreater(float(samples.min()), 2)
        self.assertLess(float(samples.max()), 6)
        weighted = sample_pdf(edges, torch.tensor([[0.0, 10.0, 0.0]]), 32, False)
        self.assertTrue(((weighted >= 3) & (weighted <= 4)).all())

    def test_compositing_constant_density_matches_analytic_transmission(self):
        positions = torch.tensor([[1.0, 2.0, 3.0]])
        rgb, weights, _ = composite(torch.ones_like(positions), torch.zeros(1, 3, 3), positions,
                                     torch.tensor([0.5]), torch.tensor([3.5]), 1.0)
        torch.testing.assert_close(rgb, torch.full((1, 3), np.exp(-3), dtype=torch.float32))
        torch.testing.assert_close(weights.sum(-1), torch.tensor([1 - np.exp(-3)], dtype=torch.float32))

    def test_nested_camera_selection_is_unique(self):
        poses = np.broadcast_to(np.eye(4), (12, 4, 4)).copy()
        poses[:, 0, 3] = np.linspace(-1, 1, 12)
        order = reference_order(poses)
        self.assertEqual(len(order), 8)
        self.assertEqual(len(set(order)), 8)
        self.assertEqual(order[0], 0)


if __name__ == "__main__":
    unittest.main()
