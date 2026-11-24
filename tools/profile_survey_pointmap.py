"""Paired geometry or source-to-target RGB timings after warm-up."""
import argparse
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image

from relation_nvs.io import load_camera,sha256,write_json
from relation_nvs.survey_pointmap import calibrated_pair_depth,load_pair_model,ordered_pointmaps,pointmap_precision
from tools.benchmark_stereo_transport import OUT,ROOT,read,raster
from tools.benchmark_survey_global_alignment import optimize
from relation_nvs.stereo_transport import SurveyStereo
from relation_nvs.survey_flow import estimate_flow, get_flow_model
from relation_nvs.survey_visibility import render_surfaces, combine_surface_and_inverse


@pointmap_precision()
def main(render_rgb=False,source_fit=False,group='syndrone_transfer'):
    model=load_pair_model()
    matcher=get_flow_model() if render_rgb else None
    cv2.setNumThreads(2)
    folder=OUT/group
    rows=[]
    for i,row in enumerate(read(folder/'predictions.json')):
        cams=[load_camera(c) for c in row['cameras'][:2]]
        target=load_camera(row['cameras'][2])
        paths=[ROOT/p for p in row['paths'][:2]]
        if [sha256(p) for p in paths]!=row['source_hashes']:
            raise ValueError('Source changed')
        ims=[raster(p,c) for p,c in zip(paths,cams)]
        # Own-ray fitting does not consume optical flow. All downstream rendering is shared.
        def ours():
            pair=ordered_pointmaps(model,ims)
            depths=calibrated_pair_depth(pair,cams,None,'rays')[0]
            if source_fit:
                return [(pair[j][0][j][...,2],d) for j,d in enumerate(depths)]
            return depths
        def upstream():
            torch.manual_seed(123)
            np.random.seed(123)
            return optimize(ims,cams,model)
        def run(fn):
            depths=fn()
            if len(depths)!=2:
                raise ValueError('Missing depth')
            if render_rgb:
                rule='source_fit' if source_fit and fn is ours else 'metric_pair'
                atlas=SurveyStereo(ims,cams,estimate_flow(matcher,ims),depths,rule)
                inverse,_=atlas.render(target)
                _,surface,_=render_surfaces(atlas,target)
                return combine_surface_and_inverse(surface,inverse)[0]

        if i==0:
            for fn in (ours,upstream):
                run(fn)
        result={}
        ops=[('rays',ours),('optimizer',upstream)]
        for name,fn in (ops if i%2==0 else ops[::-1]):
            torch.cuda.synchronize()
            start=time.perf_counter()
            rgb=run(fn)
            torch.cuda.synchronize()
            result[name]=time.perf_counter()-start
            if render_rgb:
                candidate='pointmap_source_fit.png' if source_fit else 'pointmap_rays.png'
                prediction=folder/row['case']/(candidate if name=='rays' else 'pointmap_global_optimizer.png')
                if not np.array_equal(rgb,np.asarray(Image.open(prediction))):
                    raise ValueError(f'Timed RGB differs from the scored output at {prediction}')
        rows.append(dict(case=row['case'],seconds=result))
    scope='From images and loaded weights through both metric depth maps'
    if render_rgb:
        scope='From images and loaded weights through full two-source RAFT, scene construction and one target RGB'
    name='rgb_timing.json' if render_rgb else 'geometry_timing.json'
    if source_fit:
        name='source_fit_rgb_timing.json'
    write_json(folder/name,dict(device=torch.cuda.get_device_name(),
               omp_threads=os.environ.get('OMP_NUM_THREADS'),openblas_threads=os.environ.get('OPENBLAS_NUM_THREADS'),
               candidate='pointmap_source_fit' if source_fit else 'pointmap_rays',
               scope=scope+'. Excludes file I/O and weight loading. Alternating order. One warm-up pair.',
               rows=rows,median_seconds={m:np.median([r['seconds'][m] for r in rows]) for m in ('rays','optimizer')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render-rgb',action='store_true')
    parser.add_argument('--source-fit',action='store_true',help='Time the full pipeline with source-only shape selection')
    parser.add_argument('--group',choices=('syndrone_transfer','mill19_rubble_repeat'),default='syndrone_transfer')
    args=parser.parse_args()
    main(args.render_rgb or args.source_fit,args.source_fit,args.group)
