"""Score delivered source geometry against the simulator's axial-depth buffers."""
import cv2
import numpy as np
from PIL import Image

from relation_nvs.io import load_camera, sha256, write_json
from relation_nvs.stereo_transport import grid_for
from tools.benchmark_stereo_transport import OUT, ROOT, PAIR_METHODS, read
from tools.benchmark_survey_pointmaps import DIAGNOSTIC, DEPTH_CONTROLS


def depth_masks(truth, peer, source, other):
    valid = np.isfinite(truth) & (truth > 0)
    xy = grid_for(source)
    rays = np.dstack((xy,np.ones_like(truth))) @ np.linalg.inv(source.intrinsics).T
    world = (rays * np.where(valid,truth,0)[...,None]) @ source.camera_to_world[:3,:3].T + source.camera_to_world[:3,3]
    local = (world-other.camera_to_world[:3,3]) @ other.camera_to_world[:3,:3]
    uvz = local @ other.intrinsics.T
    uv = (uvz[...,:2]/np.maximum(local[...,2:],1e-6)).astype(np.float32)
    inside = (local[...,2]>0) & ((uv>=0)&(uv<=[other.width-1,other.height-1])).all(-1)
    sampled = cv2.remap(peer,uv,None,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    visible = inside & (sampled>0) & (abs(local[...,2]-sampled) <= .05*sampled)
    kernel = np.ones((3,3),np.uint8)
    low = cv2.erode(np.where(valid,truth,np.inf),kernel)
    high = cv2.dilate(np.where(valid,truth,0),kernel)
    return dict(all_valid=valid,peer_visible=valid&visible,other=valid&~visible,boundary=valid&(high>1.10*low.astype(np.float64)))


def main():
    folder = OUT / DIAGNOSTIC
    methods = PAIR_METHODS + DEPTH_CONTROLS
    rows = []
    manifests = {m:read(folder/(m+'.json')) for m in methods}
    for row in read(folder/'predictions.json'):
        cams = [load_camera(c) for c in row['cameras'][:2]]
        source = ROOT/row['paths'][0]
        refs = read(source.parent/'source_to_height50.json')['references']
        truths = []
        for ref,c in zip(refs,cams):
            path = source.parent/ref['depth']
            if sha256(path)!=ref['hashes']['depth'] or ref['depth_kind']!='z':
                raise ValueError('Source depth reference changed')
            truths.append(np.asarray(Image.fromarray(np.load(path)).resize((c.width,c.height),Image.Resampling.NEAREST)))
        for method in methods:
            path = folder/row['case']/(method+'_depth.npz')
            if sha256(path)!=manifests[method][row['case']]['depth_sha256']:
                raise ValueError('Rendered geometry changed')
            with np.load(path) as data:
                depths = data['depth']
            for j,(truth,c,depth) in enumerate(zip(truths,cams,depths)):
                for mask_name,mask in depth_masks(truth,truths[1-j],c,cams[1-j]).items():
                    usable = mask & np.isfinite(depth) & (depth>0)
                    if not usable.any():
                        raise ValueError('Empty evaluation region')
                    ratio = depth[usable]/truth[usable]
                    rows.append(dict(case=row['case'],source=c.camera_id,method=method,mask=mask_name,
                                     truth_sha256=refs[j]['hashes']['depth'],pixels=int(mask.sum()),
                                     fraction=float(mask.sum()/np.isfinite(truth).sum()),
                                     coverage=float(usable.sum()/mask.sum()),abs_rel=float(np.mean(abs(ratio-1))),
                                     delta_1_25=float(np.mean(np.maximum(ratio,1./ratio)<1.25))))
    means = {m:{} for m in methods}
    for m in methods:
        for region in ('all_valid','peer_visible','other','boundary'):
            sub = [r for r in rows if r['method']==m and r['mask']==region]
            means[m][region] = {k:float(np.mean([r[k] for r in sub])) for k in ('abs_rel','delta_1_25','fraction','coverage')}
    write_json(folder/'source_depth_scores.json',dict(rows=rows,means=means))


if __name__=='__main__':
    main()
