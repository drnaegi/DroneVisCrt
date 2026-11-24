"""Compare a pose-conditioned metric geometry model to the common pointmap-ray correction."""
import argparse
import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image
import torch
from safetensors.torch import load_model as load_safetensors_model

from relation_nvs.io import load_camera,sha256,write_json
from relation_nvs.stereo_transport import SurveyStereo
from relation_nvs.survey_pointmap import calibrated_pair_depth,pointmap_precision
from relation_nvs.survey_flow import get_flow_model,estimate_flow
from relation_nvs.survey_visibility import render_surfaces,combine_surface_and_inverse
from tools.benchmark_stereo_transport import OUT,ROOT,read,raster,score

EXTERNAL=Path('/autodl-tmp/DroneVisCrt')
CODE=EXTERNAL/'code/map-anything-3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9'
DINO=EXTERNAL/'code/dinov2-7764ea0f912e53c92e82eb78a2a1631e92725fc8'
CHECKPOINT=EXTERNAL/'checkpoints/MapAnything'
METHODS=('mapanything_metric','mapanything_rays')


def load_model():
    sys.path.insert(0,str(CODE))
    from mapanything.models import MapAnything
    path=CHECKPOINT/'model.safetensors'
    if sha256(path)!='981f060c64664dff3272b5f5a823d350abe71a2f144444db4cfc325f3ed5a3a0':
        raise ValueError('Unexpected MapAnything release')
    upstream_load=torch.hub.load
    def hub(repo,model,**kwargs):
        if 'dinov2' not in str(repo) or model!='dinov2_vitg14':
            raise ValueError((repo,model))
        return upstream_load(str(DINO),model,source='local',pretrained=False)
    with patch.object(torch.hub,'load',hub):
        model=MapAnything(**read(CHECKPOINT/'config.json'))
    load_safetensors_model(model,str(path),strict=True)
    return model.cuda().eval()


@torch.inference_mode()
def geometries(model,images,cameras):
    from mapanything.utils.image import preprocess_inputs
    w,h=cameras[0].width,cameras[0].height
    size=tuple(14*round(x/14) for x in (w,h))
    views=[]
    for im,c in zip(images,cameras):
        views.append(dict(img=Image.fromarray(im).resize(size,Image.Resampling.LANCZOS),
                          intrinsics=c.resized(*size).intrinsics.astype(np.float32),
                          camera_poses=c.camera_to_world.astype(np.float32),is_metric_scale=True))
    inputs=preprocess_inputs(views,resize_mode='fixed_size',size=size)
    for inp,view in zip(inputs,views):
        if not np.allclose(inp['intrinsics'].numpy()[0],view['intrinsics'],atol=1e-3):
            raise ValueError('Preprocessing unexpectedly changed the source camera')
    pred=model.infer(inputs,memory_efficient_inference=True,use_amp=True,amp_dtype='bf16',apply_mask=True,
                     mask_edges=True,apply_confidence_mask=False,use_multiview_confidence=False)
    def field(p,key):
        return p[key][0].detach().float().cpu().numpy()
    depths,maps,confidence,valid=[],[],[],[]
    reference=field(pred[0],'camera_poses')
    for p in pred:
        raw=field(p,'depth_z').squeeze(-1)
        mask=np.isfinite(raw)&(raw>0)
        nearest=cv2.resize(mask.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST).astype(bool)
        mass=cv2.resize(mask.astype(np.float32),(w,h),interpolation=cv2.INTER_LINEAR)
        def resize(v):
            good=mask if v.ndim==2 else mask[...,None]
            denom=mass if v.ndim==2 else mass[...,None]
            return cv2.resize(np.where(good,v,0.),(w,h),interpolation=cv2.INTER_LINEAR)/np.maximum(denom,1e-8)
        depths.append(np.where(nearest,resize(raw),np.nan))
        world=field(p,'pts3d')
        q=(world-reference[:3,3])@reference[:3,:3]
        maps.append(resize(q))
        confidence.append(np.where(nearest,resize(field(p,'conf').reshape(raw.shape)),1.))
        valid.append(nearest)
    rays,_=calibrated_pair_depth([(maps,confidence)],cameras,None,'rays')
    return (depths,[np.where(ok,d,np.nan) for ok,d in zip(valid,rays)])


@pointmap_precision()
def run(groups, scale_control=False):
    model,matcher=load_model(),get_flow_model()
    for group in groups:
        folder=OUT/group
        methods = ('mapanything_median',) if scale_control else METHODS
        records={name:{} for name in methods}
        for row in read(folder/'predictions.json'):
            cams=[load_camera(c) for c in row['cameras']]
            paths=[ROOT/p for p in row['paths'][:2]]
            if [sha256(p) for p in paths]!=row['source_hashes']:
                raise ValueError('Source bytes changed')
            images=[raster(p,c) for p,c in zip(paths,cams)]
            flow=estimate_flow(matcher,images)
            depths=geometries(model,images,cams[:2])
            for name,depth in zip(methods,depths):
                atlas=SurveyStereo(images,cams[:2],flow,depth,'median_stereo' if scale_control else 'metric_pair')
                inverse,_=atlas.render(cams[2])
                _,blend,_=render_surfaces(atlas,cams[2])
                rgb,_=combine_surface_and_inverse(blend,inverse)
                path=folder/row['case']/(name+'.png')
                Image.fromarray(rgb).save(path)
                records[name][row['case']]=dict(rgb_sha256=sha256(path),valid_depth=[float(np.isfinite(d).mean()) for d in depth])
            print(group,row['case'],flush=True)
        for name,values in records.items():
            write_json(folder/(name+'.json'),values)
        score(group,only=methods,output='mapanything_scale_scores.json' if scale_control else 'mapanything_scores.json')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('groups',nargs='+')
    p.add_argument('--scale-control',action='store_true')
    cv2.setNumThreads(2)
    args=p.parse_args()
    run(args.groups,args.scale_control)
