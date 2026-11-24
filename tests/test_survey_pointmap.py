import unittest
import numpy as np
import torch

from relation_nvs.survey_pointmap import ordered_pointmaps,ray_alignment,surface_charts
from relation_nvs.stereo_transport import grid_for,project
from test_survey_transport import camera


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.ones(1))

    def forward(self,a,b):
        def prediction(im,tag,name):
            values = (im.mean(dim=1)+1.)*10+tag
            return {name:values[...,None].repeat(1,1,1,3),'conf':values+5}
        return prediction(a['img'],100,'pts3d'),prediction(b['img'],200,'pts3d_in_other_view')


class TestSurveyPointmap(unittest.TestCase):
    def test_reversed_prediction_stays_on_original_pixel_grid(self):
        images=[np.zeros((32,48,3),np.uint8),np.full((32,48,3),255,np.uint8)]
        first,second=ordered_pointmaps(FakeModel(),images)
        for maps,expected in [(first,(100,220)),(second,(200,120))]:
            for j in range(2):
                np.testing.assert_allclose(maps[0][j],expected[j],atol=1e-5)
                np.testing.assert_allclose(maps[1][j],expected[j]+5,atol=1e-5)

    def test_known_scale_and_regularized_affine_ray_fit(self):
        rng=np.random.default_rng(1234)
        q=rng.uniform([-1,-1,1.2],[1,1,2.5],(2400,3))
        c=np.zeros_like(q)
        c[::2,0]=3
        weight=np.ones(len(q))
        metric=3*q
        def rays(values):
            v=values-c
            return v/np.linalg.norm(v,axis=1,keepdims=True)
        scale=ray_alignment(q,rays(metric),c,weight,False)
        np.testing.assert_allclose(scale,np.c_[3*np.eye(3),np.zeros(3)],atol=1e-8)
        from scipy.spatial.transform import Rotation
        rot=Rotation.from_rotvec([.06,-.08,.03]).as_matrix()
        rotated=3*q@rot.T+[.2,-.1,.1]
        sim=ray_alignment(q,rays(rotated),c,weight,False,similarity=True)
        np.testing.assert_allclose(np.linalg.svd(sim[:,:3],compute_uv=False),3.,atol=.05)
        self.assertLess(np.mean(np.linalg.norm(q@sim[:,:3].T+sim[:,3]-rotated,axis=1)),.03)
        a=np.array([[3.1,.2,.03],[.05,2.8,.12],[.12,.04,3.2]])
        actual=q@a.T+[.1,.04,-.2]
        transform=ray_alignment(q,rays(actual),c,weight,True)
        fitted=q@transform[:,:3].T+transform[:,3]
        self.assertLess(np.mean(np.linalg.norm(actual-fitted,axis=1)),.05)
        self.assertIsNone(ray_alignment(q,rays(metric),0*c,weight,True))
        self.assertIsNone(ray_alignment(q,rays(metric),c,0*weight,True))

    def test_corrected_shape_validated_on_disjoint_source_tiles(self):
        from relation_nvs.survey_pointmap_fit import fit_pointmap_depth

        rng=np.random.default_rng(781)
        own=np.exp(rng.normal(2,.25,(160,160)))
        rays=np.exp(rng.normal(2,.25,own.shape))
        plane=np.full_like(own,9.)
        reliable=np.ones_like(own,dtype=bool)
        target=1.15*own**.65*rays**.35
        depth,choice,values=fit_pointmap_depth(plane,target,reliable,own,rays)
        self.assertEqual(choice,'mixture')
        self.assertAlmostEqual(values['ray_weight'],.35,places=5)
        np.testing.assert_allclose(depth,target)
        reliable[:]=False
        sparse,choice,_=fit_pointmap_depth(plane,target,reliable,own,rays)
        self.assertEqual(choice,'rays')
        np.testing.assert_allclose(sparse,rays)
        corrected,_,_=fit_pointmap_depth(plane,1.2*rays,~reliable,own,rays,scale_only=True)
        np.testing.assert_allclose(corrected,1.2*rays)

    def test_surface_chart_reproduces_peer_projection(self):
        cams=[camera('a',0,0),camera('b',1,0)]
        depth=np.full((96,96),5.)
        charts=surface_charts([depth,depth],cams)
        for j,c in enumerate(cams):
            xy=grid_for(c)
            dirs=np.dstack((xy,np.ones_like(depth)))@np.linalg.inv(c.intrinsics).T
            xyz=c.camera_to_world[:3,3]+depth[...,None]*dirs@c.camera_to_world[:3,:3].T
            np.testing.assert_allclose(charts[j],project(xyz,cams[1-j]),atol=1e-5)


if __name__=='__main__':
    unittest.main()
