"""RIFE reference for targets lying between both source positions along the flight direction."""
import argparse
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from tools.benchmark_stereo_transport import OUT,ROOT,raster,read
from relation_nvs.io import load_camera,sha256,write_json
from relation_nvs.aerial_mosaic import AerialMosaic


def run(groups, align):
    root=ROOT/'checkpoints/RIFE'
    pkg=root/'RIFEv4.26_0921'
    sys.path[:0]=[str(root/'practical'),str(pkg)]
    from IFNet_HDv3 import IFNet
    weights=pkg/'flownet.pkl'
    if sha256(weights)!='45c7f74156704769dc9f85cfcaf8552e1e926f9399dcfa3a553dee88fac6f53f':
        raise ValueError('Unexpected RIFE weights')
    model=IFNet()
    states=torch.load(weights,weights_only=True,map_location='cpu')
    state = {k.removeprefix('module.'):v for k,v in states.items()}
    state = {k:v for k,v in state.items() if not k.startswith(('teacher.','caltime.'))}
    model.load_state_dict(state,strict=True)
    model=model.cuda().eval()
    for group in groups:
        folder=OUT/group
        records={}
        excluded = {}
        for r in read(folder/'predictions.json'):
            cs=[load_camera(c) for c in r['cameras']]
            centers=np.stack([c.camera_to_world[:3,3] for c in cs])
            b=centers[1]-centers[0]
            frac=float((centers[2]-centers[0])@b / max(b@b,1e-9))
            if not 0<frac<1:
                excluded[r['case']] = dict(fraction=frac, reason='outside open interpolation interval')
                continue
            frames=[raster(ROOT/p,c) for p,c in zip(r['paths'][:2],cs)]
            if align:
                fallback, _, coords = AerialMosaic.from_views(frames,cs[:2]).render(cs[2])
                masks = [((uv >= 0) & (uv <= [c.width-1,c.height-1])).all(axis=-1)
                         for uv,c in zip(coords,cs)]
                frames = [cv2.remap(im,uv,None,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
                          for im,uv in zip(frames,coords)]
            rgb=torch.from_numpy(np.concatenate(frames,axis=-1)).cuda().permute(2,0,1)[None].float()/255.
            h,w=rgb.shape[-2:]
            padded=F.pad(rgb,(0,(-w)%64,0,(-h)%64))
            with torch.inference_mode():
                rgb=model(padded,frac,[16,8,4,2,1])[2][-1][0,:,:h,:w]
            result=rgb.clamp(0,1).mul(255).round().byte().permute(1,2,0).cpu().numpy()
            name = 'rife_ground' if align else 'rife'
            if align:
                result = np.where((masks[0] & masks[1])[...,None],result,fallback)
            output=folder/r['case']/(name+'.png')
            Image.fromarray(result).save(output)
            records[r['case']]=dict(rgb_sha256=sha256(output),fraction=frac)
        name = 'rife_ground' if align else 'rife'
        write_json(folder/(name+'.json'),records)
        write_json(folder/(name+'_eligibility.json'),dict(eligible=list(records),excluded=excluded))
        print(group,len(records),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('groups',nargs='+')
    p.add_argument('--align',action='store_true')
    args = p.parse_args()
    run(args.groups,args.align)
