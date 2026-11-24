import unittest

import numpy as np

from relation_nvs.aerial_mosaic import AerialMosaic
from relation_nvs.geometry import Camera


def camera(x):
    pose = np.diag([1., -1., -1., 1.])
    pose[:3, 3] = [x, 0, 1]
    return Camera(str(x), [[32, 0, 15.5], [0, 32, 15.5], [0, 0, 1]], pose, 32, 32)


class AerialMosaicTest(unittest.TestCase):
    def test_identity_and_unsupported_region(self):
        images = np.random.default_rng(24).integers(0, 256, (2, 32, 32, 3), dtype=np.uint8)
        sources = [camera(0), camera(2)]
        model = AerialMosaic.from_views(images, sources)
        for index, source in enumerate(sources):
            rgb, provenance, coord = model.render(source)
            np.testing.assert_array_equal(rgb, images[index])
            self.assertTrue(np.all(provenance == index + 1))
        rgb, provenance, _ = model.render(camera(5))
        self.assertTrue(np.any(rgb))
        self.assertFalse(np.any(provenance))

    def test_horizon_is_rejected(self):
        pose = np.eye(4)
        pose[2, 3] = 1
        c = Camera('up', np.eye(3), pose, 32, 32)
        with self.assertRaisesRegex(ValueError, 'look down'):
            AerialMosaic.from_views([np.zeros((32, 32, 3), np.uint8)] * 2, [c, c])

    def test_dji_half_turn_and_reverse(self):
        from relation_nvs.aerial_mosaic import nadir_camera
        c = camera(2)
        corrected = nadir_camera(c, 179.9)
        np.testing.assert_array_equal(corrected.camera_to_world[:2,:2], -c.camera_to_world[:2,:2])
        np.testing.assert_array_equal(corrected.camera_to_world[:3,3], c.camera_to_world[:3,3])
        self.assertIs(nadir_camera(c), c)
        self.assertIs(nadir_camera(c, 1.0), c)

    def test_featureless_sources_keep_metric_camera_positions(self):
        from relation_nvs.aerial_mosaic import SurveyCalibration
        images = [np.zeros((32,32,3),np.uint8)] * 2
        fit = SurveyCalibration.estimate(images,[camera(0),camera(2)])
        self.assertEqual(fit.inliers, 0)
        np.testing.assert_array_equal(fit.apply(camera(4)).camera_to_world, camera(4).camera_to_world)
