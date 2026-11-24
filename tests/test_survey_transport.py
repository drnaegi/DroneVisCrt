import unittest

import numpy as np

from relation_nvs.geometry import Camera
from relation_nvs.survey_bands import combine
from relation_nvs.stereo_transport import SurveyStereo, grid_for, project, rectify_source_pair, unrectify_flow, select_depth_prior, log_depth_sensitivity


def camera(name, x, y):
    pose = np.diag([1., -1., -1., 1.])
    pose[:3, 3] = [x, y, 10.]
    return Camera(name, [[80.,0,47.5],[0,80.,47.5],[0,0,1.]], pose, 96,96)


def translations(cams, depth):
    return [np.broadcast_to([(cams[j].camera_to_world[0,3] - cams[1-j].camera_to_world[0,3])*80/depth, 0.],
                            (96,96,2)).astype(np.float32) for j in range(2)]


class TestSurveyTransport(unittest.TestCase):
    def setUp(self):
        self.cams = [camera('a',0,0), camera('b',1,0)]
        grid = grid_for(self.cams[0])
        self.image = np.stack([grid[...,0], grid[...,1], grid[...,1]],axis=-1).astype(np.uint8)
        self.images = [self.image, np.roll(self.image,-16,axis=1)]

    def test_triangulated_elevated_surface_and_transverse_target(self):
        # The true scene lies halfway between the cameras and the nominal ground.
        atlas = SurveyStereo(self.images, self.cams, translations(self.cams,5))
        target = camera('t',0,1)
        _, xyz, _, _, _ = atlas.geometry[0]
        np.testing.assert_allclose(xyz[40,40], [-.46875,.46875,5.], atol=1e-6)
        np.testing.assert_allclose(project(xyz[40,40],target), [40.,56.], atol=1e-5)
        actual,_ = atlas.render(target)
        np.testing.assert_array_equal(actual[56,40], self.image[40,40])
        plane,_ = atlas.render(target,'flow_plane')
        self.assertGreater(abs(int(plane[56,40,1])-40),5)

    def test_identity_with_consistent_matches(self):
        atlas = SurveyStereo(self.images,self.cams,translations(self.cams,5))
        for j,cam in enumerate(self.cams):
            rgb,_ = atlas.render(cam)
            np.testing.assert_array_equal(rgb[20:70,20:70],self.images[j][20:70,20:70])

    def test_log_depth_projection_derivative(self):
        source = self.cams[0]
        origin = source.camera_to_world[:3,3]
        rays = np.array([[.2,-.3,-1.],[-.2,.1,-1.]])
        target = camera('offset',1.,1.5)
        step = .00001
        xyz = origin + rays*5.
        left = project(origin+np.exp(-step)*(xyz-origin), target)
        right = project(origin+np.exp(step)*(xyz-origin), target)
        np.testing.assert_allclose(log_depth_sensitivity(xyz,source,target),(right-left)/(2*step),atol=1e-7)
        np.testing.assert_allclose(log_depth_sensitivity(xyz,source,source), 0.,atol=1e-10)

    def test_precision_identity_and_missing_prior(self):
        invalid = np.full((96,96), np.nan)
        atlas = SurveyStereo(self.images,self.cams,translations(self.cams,5),
                             monocular=[invalid,invalid],prior_rule='precision')
        self.assertTrue(all(np.isfinite(v).all() for v in atlas.depth_variance))
        self.assertLess(atlas.depth_variance[0][40,40], atlas.depth_variance[0][40,0])
        rgb,_ = atlas.render(self.cams[0],precision_donor=True)
        np.testing.assert_array_equal(rgb[20:70,20:70],self.images[0][20:70,20:70])

    def test_band_coefficients_recover_each_original_render(self):
        cams = [*self.cams,camera('off',.4,.5)]
        for weight,expected in [(0.,self.images[0]),(1.,self.images[1])]:
            result = combine(*self.images,cams,[[weight]]*3)
            np.testing.assert_array_equal(result,expected)

    def test_rectification_maps_zero_residual_to_ground_flow(self):
        _, homographies = rectify_source_pair(self.images,self.cams)
        forward = unrectify_flow(np.zeros((2,96,96,2),np.float32), self.cams, homographies)
        np.testing.assert_allclose(forward, translations(self.cams,10), atol=1e-5)

    def test_depth_model_selection_on_spatially_withheld_matches(self):
        y,x = np.indices((192,192))
        plane = np.ones(x.shape)*10
        stereo = plane*np.exp(.15*np.sin(x/12)+.06*np.cos(y/9))
        shape, model, _ = select_depth_prior(plane,stereo,2*stereo,np.ones(x.shape,bool))
        self.assertEqual(model,'mono_shape')
        np.testing.assert_allclose(shape,stereo,rtol=.001)
        shape, model, _ = select_depth_prior(plane,stereo,2*stereo,np.zeros(x.shape,bool))
        np.testing.assert_array_equal(shape,plane)
        self.assertEqual(model,'plane')
