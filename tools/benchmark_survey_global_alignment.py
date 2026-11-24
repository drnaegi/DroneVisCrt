"""Official two-view DUSt3R known-pose optimizer evaluated in the common renderer."""
import argparse
import time

import cv2
import numpy as np
from PIL import Image
import torch

from relation_nvs.io import load_camera,sha256,write_json
from relation_nvs.stereo_transport import SurveyStereo
from relation_nvs.survey_flow import get_flow_model,estimate_flow
from relation_nvs.survey_pointmap import load_pair_model
from relation_nvs.survey_visibility import render_surfaces,combine_surface_and_inverse
from tools.benchmark_stereo_transport import OUT,ROOT,read,raster,score

NAME = 'pointmap_global_optimizer'


def optimize(images,cameras,model,steps=300):
    from dust3r.inference import inference
    from dust3r.image_pairs import make_pairs
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

    h,w = images[0].shape[:2]
    if max(h,w)!=512:
        raise ValueError('Optimizer study expects a 512-pixel long edge')
    shape = tuple(16*((n+15)//16) for n in (w,h))
    left,top = (shape[0]-w)//2,(shape[1]-h)//2
    pad = ((top,shape[1]-h-top),(left,shape[0]-w-left),(0,0))
    resized = [np.pad(im,pad,mode='edge') for im in images]
    views = [dict(img=torch.from_numpy(im.copy()).permute(2,0,1)[None].float()/127.5-1,
                  true_shape=np.array([shape[::-1]],np.int32),idx=i,instance=str(i)) for i,im in enumerate(resized)]
    predictions = inference(make_pairs(views,scene_graph='complete',symmetrize=True),model,'cuda',batch_size=1)
    scene = global_aligner(predictions,device='cuda',mode=GlobalAlignerMode.PointCloudOptimizer,optimize_pp=True)
    intrinsics = np.stack([c.intrinsics for c in cameras])
    intrinsics[:,0,2] += left
    intrinsics[:,1,2] += top
    if not np.allclose(intrinsics[:,0,0],intrinsics[:,1,1],rtol=1e-5):
        raise ValueError('This optimizer control assumes a common x/y focal in each view')
    poses = np.stack([c.camera_to_world for c in cameras])
    scene.preset_pose(poses)
    scene.preset_focal(list(intrinsics[:,0,0]))
    scene.preset_principal_point(intrinsics[:,:2,2])
    scene.compute_global_alignment(init='known_poses',niter=steps,schedule='linear',lr=.01)
    with torch.no_grad():
        if not (np.allclose(scene.get_im_poses().cpu().numpy(),poses,atol=.001) and
                np.allclose(scene.get_intrinsics().cpu().numpy(),intrinsics,atol=.001)):
            raise ValueError('Source calibration changed')
        depths = torch.stack(scene.get_depthmaps()).cpu().numpy()
    return depths[:,top:top+h,left:left+w]


from relation_nvs.survey_pointmap import pointmap_precision


@pointmap_precision()
def generate(group,steps=300):
    name = NAME if steps==300 else f'pointmap_optimizer_{steps}'
    folder = OUT/group
    model,matcher = load_pair_model(),get_flow_model()
    records = {}
    for row in read(folder/'predictions.json'):
        cams = [load_camera(c) for c in row['cameras']]
        sources = [ROOT/p for p in row['paths'][:2]]
        if [sha256(p) for p in sources]!=row['source_hashes']:
            raise ValueError('Source file changed')
        images = [raster(p,c) for p,c in zip(sources,cams)]
        torch.manual_seed(123)
        np.random.seed(123)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        depth = optimize(images,cams[:2],model,steps)
        torch.cuda.synchronize()
        seconds = time.perf_counter()-t0
        atlas = SurveyStereo(images,cams[:2],estimate_flow(matcher,images),depth,'metric_pair')
        inverse,_ = atlas.render(cams[2])
        _,blend,_ = render_surfaces(atlas,cams[2])
        rgb,_ = combine_surface_and_inverse(blend,inverse)
        path = folder/row['case']/(name+'.png')
        Image.fromarray(rgb).save(path)
        records[row['case']] = dict(rgb_sha256=sha256(path),optimizer_seconds=seconds)
        print(group,row['case'],flush=True)
    write_json(folder/(name+'.json'),records)
    result='global_optimizer_scores.json' if steps==300 else f'optimizer_{steps}_scores.json'
    score(group,only=(name,),output=result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('groups',nargs='+')
    parser.add_argument('--steps',type=int,choices=(1,10,30,300),default=300)
    cv2.setNumThreads(2)
    args=parser.parse_args()
    for group in args.groups:
        generate(group,args.steps)
